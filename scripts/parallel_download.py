"""Bounded parallel HTTP ranges, preserving an existing curl partial download."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

import requests


def download(url, output, workers=8, chunk_bytes=8 * 1024 * 1024):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.head(url, allow_redirects=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers['Content-Length'])
        identity = {'url': url, 'bytes': total, 'etag': response.headers.get('ETag')}
    metadata = output.with_name(output.name + '.source.json')
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError('Remote file changed; refusing to mix download versions')
    metadata.write_text(json.dumps(identity, indent=2) + '\n')
    offset = output.stat().st_size if output.exists() else 0
    if offset > total:
        raise ValueError('Local partial file is larger than the remote file')
    parts = output.with_name(output.name + '.chunks')
    parts.mkdir(exist_ok=True)

    def fetch(bounds):
        first, last = bounds
        size = last - first + 1
        part = parts / f'{first}-{last}'
        if part.exists() and part.stat().st_size == size:
            return part
        temporary = part.with_suffix('.tmp')
        for attempt in range(5):
            try:
                headers = {'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'}
                if identity['etag']:
                    headers['If-Match'] = identity['etag']
                with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as r:
                    r.raise_for_status()
                    expected = f'bytes {first}-{last}/{total}'
                    if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not return the exact requested byte range')
                    with temporary.open('wb') as target:
                        written = 0
                        for block in r.iter_content(256 * 1024):
                            written += len(block)
                            if written > size:
                                raise ValueError('Server returned too many bytes')
                            target.write(block)
                if temporary.stat().st_size != size:
                    raise ValueError('Incomplete byte range')
                temporary.replace(part)
                return part
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    started, initial = time.monotonic(), offset
    bounds = iter((i, min(i + chunk_bytes, total) - 1) for i in range(offset, total, chunk_bytes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = []
        for _ in range(workers):
            item = next(bounds, None)
            if item is not None:
                pending.append((item, executor.submit(fetch, item)))
        with output.open('ab') as target:
            while pending:
                (first, last), future = pending.pop(0)
                part = future.result()
                assert target.tell() == first
                with part.open('rb') as source:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                target.flush()
                os.fsync(target.fileno())
                part.unlink()
                offset = last + 1
                rate = (offset - initial) / max(time.monotonic() - started, 0.001)
                eta = (total - offset) / max(rate, 1)
                print(f'{100 * offset / total:.1f}% | {rate / 1024**2:.2f} MiB/s | '
                      f'ETA {eta / 3600:.1f} hours', flush=True)
                item = next(bounds, None)
                if item is not None:
                    pending.append((item, executor.submit(fetch, item)))
    if output.stat().st_size != total:
        raise ValueError('Downloaded file size does not match remote size')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('output')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    download(args.url, args.output, args.workers)


if __name__ == '__main__':
    main()
