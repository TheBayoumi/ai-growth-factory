import {bundle} from '@remotion/bundler';
import path from 'node:path';

type BundleProjectOptions = {
  projectRoot: string;
  publicDir?: string;
  enableCaching?: boolean;
  onProgress?: (progress: number) => void;
};

/**
 * Bundle the TypeScript Remotion source graph while preserving NodeNext-compatible
 * `.js` import specifiers for the emitted command-line modules.
 *
 * TypeScript intentionally emits ESM imports such as `./Root.js`, while Remotion's
 * webpack build consumes the original `Root.tsx`. Webpack's extension alias maps only
 * those explicit JavaScript specifiers back to their source equivalents.
 */
export const bundleRemotionProject = async ({
  projectRoot,
  publicDir,
  enableCaching = true,
  onProgress,
}: BundleProjectOptions): Promise<string> => {
  return bundle({
    entryPoint: path.join(projectRoot, 'src', 'index.ts'),
    rootDir: projectRoot,
    ...(publicDir ? {publicDir} : {}),
    enableCaching,
    onProgress,
    webpackOverride: (configuration) => ({
      ...configuration,
      resolve: {
        ...configuration.resolve,
        extensionAlias: {
          ...(configuration.resolve?.extensionAlias ?? {}),
          '.js': ['.js', '.ts', '.tsx'],
        },
      },
    }),
  });
};
