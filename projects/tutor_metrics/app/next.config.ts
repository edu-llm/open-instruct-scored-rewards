import type { NextConfig } from 'next';
import { withBotId } from 'botid/next/config';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // `pg` opens TCP sockets, which is fine on the Node runtime but must not be traced into the
  // client or edge bundles.
  serverExternalPackages: ['pg'],
  // This app is a subdirectory of a Python repo; without a root Turbopack walks up looking for a
  // lockfile and finds an unrelated one.
  turbopack: { root: import.meta.dirname },
  agentRules: false,
};

export default withBotId(nextConfig);
