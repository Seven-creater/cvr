import { createReadStream, createWriteStream } from "node:fs";
import { access, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const root = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(
  await readFile(join(root, "download_manifest.json"), "utf8"),
);
const concurrency = 4;

async function isValidPdf(path) {
  try {
    const info = await stat(path);
    if (info.size <= 50_000) return false;
    const stream = createReadStream(path, { start: 0, end: 4 });
    let header = "";
    for await (const chunk of stream) header += chunk.toString("ascii");
    return header === "%PDF-";
  } catch {
    return false;
  }
}

async function download(item) {
  const destination = join(root, item.file);
  const partial = `${destination}.download`;
  if (await isValidPdf(destination)) {
    return { file: item.file, status: "reused", bytes: (await stat(destination)).size };
  }

  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(item.source_url, {
        redirect: "follow",
        headers: { "user-agent": "Mozilla/5.0 Audio-CVR paper collection" },
        signal: AbortSignal.timeout(180_000),
      });
      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }
      await pipeline(Readable.fromWeb(response.body), createWriteStream(partial));
      if (!(await isValidPdf(partial))) {
        throw new Error("downloaded payload is not a valid PDF");
      }
      await rename(partial, destination);
      return {
        file: item.file,
        status: "downloaded",
        bytes: (await stat(destination)).size,
      };
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, attempt * 1500));
    }
  }
  return {
    file: item.file,
    status: "failed",
    bytes: 0,
    error: String(lastError),
  };
}

const results = new Array(manifest.length);
let nextIndex = 0;

async function worker() {
  while (true) {
    const index = nextIndex;
    nextIndex += 1;
    if (index >= manifest.length) return;
    const item = manifest[index];
    process.stdout.write(`START ${item.file}\n`);
    results[index] = await download(item);
    process.stdout.write(
      `${results[index].status.toUpperCase()} ${item.file} ${results[index].bytes}\n`,
    );
  }
}

await Promise.all(Array.from({ length: concurrency }, () => worker()));
await writeFile(
  join(root, "download_results.json"),
  `${JSON.stringify(results, null, 2)}\n`,
  "utf8",
);

const failures = results.filter((item) => item.status === "failed");
process.stdout.write(
  `SUMMARY total=${results.length} valid=${results.length - failures.length} failed=${failures.length}\n`,
);
if (failures.length > 0) process.exitCode = 2;
