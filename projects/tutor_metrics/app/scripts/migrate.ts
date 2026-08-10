/**
 * Apply db/migrations/*.sql in order, once each.
 *
 *   npx dotenv -e .env.local -- npx tsx scripts/migrate.ts
 *
 * No Prisma, no drizzle-kit, no generated client. Numbered files and a schema_migrations table
 * are the whole mechanism, and every statement that matters here is hand-written SQL anyway.
 */
import fs from 'node:fs';
import path from 'node:path';
import { client } from './_pg';

const DIR = path.join(process.cwd(), 'db', 'migrations');

async function main() {
  const c = await client();
  try {
    await c.query(`create table if not exists schema_migrations (
      name text primary key, applied_at timestamptz not null default now())`);

    const applied = new Set(
      (await c.query<{ name: string }>('select name from schema_migrations')).rows.map(
        (r) => r.name,
      ),
    );
    const files = fs.readdirSync(DIR).filter((f) => f.endsWith('.sql')).sort();
    if (files.length === 0) throw new Error(`no .sql files in ${DIR}`);

    let ran = 0;
    for (const name of files) {
      if (applied.has(name)) {
        console.log(`  = ${name}`);
        continue;
      }
      const sql = fs.readFileSync(path.join(DIR, name), 'utf8');
      await c.query('begin');
      try {
        await c.query(sql);
        await c.query('insert into schema_migrations (name) values ($1)', [name]);
        await c.query('commit');
        console.log(`  + ${name}`);
        ran += 1;
      } catch (e) {
        await c.query('rollback');
        console.error(`  ! ${name}`);
        throw e;
      }
    }
    console.log(ran ? `applied ${ran} migration(s)` : 'up to date');
  } finally {
    await c.end();
  }
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
