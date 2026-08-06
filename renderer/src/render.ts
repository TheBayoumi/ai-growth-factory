import {bundle} from '@remotion/bundler';
import {ensureBrowser, renderMedia, selectComposition} from '@remotion/renderer';
import {createRequire} from 'node:module';
import {readFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseRenderSpec} from './schema.js';

const require = createRequire(import.meta.url);
const remotionPackage = require('remotion/package.json') as {version: string};

type Args = {spec: string; publicDir: string; output: string};

const parseArgs = (argv: string[]): Args => {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith('--') || !value) {
      throw new Error(`invalid renderer argument near ${key ?? '<end>'}`);
    }
    values.set(key.slice(2), value);
  }
  const spec = values.get('spec');
  const publicDir = values.get('public-dir');
  const output = values.get('output');
  if (!spec || !publicDir || !output) {
    throw new Error('required arguments: --spec, --public-dir, --output');
  }
  return {spec, publicDir, output};
};

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  const expectedVersion = process.env.REMOTION_VERSION_EXPECTED;
  if (expectedVersion && remotionPackage.version !== expectedVersion) {
    throw new Error(
      `Remotion version mismatch: installed ${remotionPackage.version}, expected ${expectedVersion}`,
    );
  }

  const specPath = path.resolve(args.spec);
  const publicDir = path.resolve(args.publicDir);
  const output = path.resolve(args.output);
  const spec = parseRenderSpec(JSON.parse(await readFile(specPath, 'utf8')));
  const configuredBrowser = process.env.REMOTION_BROWSER_EXECUTABLE?.trim();
  const browserExecutable = configuredBrowser || undefined;
  await ensureBrowser({browserExecutable, logLevel: 'warn'});
  await mkdir(path.dirname(output), {recursive: true});

  const distDir = path.dirname(fileURLToPath(import.meta.url));
  const projectRoot = path.resolve(distDir, '..');
  const serveUrl = await bundle({
    entryPoint: path.join(projectRoot, 'src', 'index.ts'),
    rootDir: projectRoot,
    publicDir,
    enableCaching: true,
    onProgress: (progress) => {
      if (progress === 100 || progress % 20 === 0) {
        console.log(`bundle:${progress}`);
      }
    },
  });
  const composition = await selectComposition({
    serveUrl,
    id: 'ShortVideo',
    inputProps: spec,
    timeoutInMilliseconds: 120_000,
    logLevel: 'warn',
    browserExecutable,
  });

  await renderMedia({
    composition,
    serveUrl,
    codec: 'h264',
    outputLocation: output,
    inputProps: spec,
    crf: 19,
    x264Preset: 'fast',
    pixelFormat: 'yuv420p',
    audioBitrate: '192K',
    imageFormat: 'jpeg',
    jpegQuality: 92,
    concurrency: process.env.REMOTION_CONCURRENCY ?? null,
    timeoutInMilliseconds: 120_000,
    overwrite: true,
    logLevel: 'warn',
    browserExecutable,
    onProgress: ({progress}) => {
      const percentage = Math.floor(progress * 100);
      if (percentage === 100 || percentage % 10 === 0) {
        console.log(`render:${percentage}`);
      }
    },
  });
  console.log(
    JSON.stringify({
      status: 'rendered',
      output,
      remotion_version: remotionPackage.version,
      duration_in_frames: spec.duration_in_frames,
      fps: spec.fps,
      width: spec.width,
      height: spec.height,
    }),
  );
};

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
