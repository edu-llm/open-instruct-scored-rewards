import { Client } from 'pg';

/** Scripts use the direct connection: migrations and bulk loads should not go via a pooler. */
export function connectionString(): string {
  const url = process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL;
  if (!url) {
    throw new Error(
      'DATABASE_URL_UNPOOLED or DATABASE_URL must be set.\n' +
        '  vercel env pull .env.local --yes   then run through: npx dotenv -e .env.local -- …',
    );
  }
  return url;
}

export async function client(): Promise<Client> {
  const url = connectionString();
  const c = new Client({
    connectionString: url,
    ssl: url.includes('sslmode=disable') || url.includes('localhost')
      ? undefined
      : { rejectUnauthorized: false },
  });
  await c.connect();
  return c;
}

export function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith('--')) {
    return process.argv[i + 1];
  }
  return fallback;
}

export function flag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}
