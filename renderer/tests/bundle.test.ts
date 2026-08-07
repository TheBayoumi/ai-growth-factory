import {mkdtemp, rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import {bundleRemotionProject} from '../src/bundle-config.js';

describe('Remotion source bundle', () => {
  it(
    'resolves NodeNext .js specifiers to TypeScript source files',
    async () => {
      const projectRoot = path.resolve(
        fileURLToPath(new URL('..', import.meta.url)),
      );
      const publicDir = await mkdtemp(
        path.join(os.tmpdir(), 'agf-remotion-public-'),
      );
      let serveUrl: string | undefined;

      try {
        serveUrl = await bundleRemotionProject({
          projectRoot,
          publicDir,
          enableCaching: false,
        });
        expect(serveUrl).toBeTruthy();
        expect(path.isAbsolute(serveUrl)).toBe(true);
      } finally {
        await rm(publicDir, {recursive: true, force: true});
        if (serveUrl && path.isAbsolute(serveUrl)) {
          await rm(serveUrl, {recursive: true, force: true});
        }
      }
    },
    120_000,
  );
});
