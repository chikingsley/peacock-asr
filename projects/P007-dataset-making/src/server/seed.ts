import { initializeDatabase } from "./db";

const result = initializeDatabase();

console.log(
  JSON.stringify(
    {
      ok: true,
      databasePath: result.databasePath,
      utteranceCount: result.utteranceCount,
    },
    null,
    2,
  ),
);
