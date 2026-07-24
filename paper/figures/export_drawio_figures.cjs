#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname);
const INPUT = path.join(ROOT, "drawio", "audio_cvr_figures.drawio");
const OUTPUT = path.join(ROOT, "generated");
const TIFF_RENDERER = path.join(ROOT, "render_drawio_tiffs.py");
const CHROME =
  process.env.CHROME_PATH ||
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const DRAWIO_LOAD_TIMEOUT_MS = Number.parseInt(
  process.env.DRAWIO_LOAD_TIMEOUT_MS || "240000",
  10,
);

const OUTPUT_NAMES = [
  "figure1_reference_confusion",
  "figure2_curation_pipeline",
];

function decodeXmlEntities(value) {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function extractPages(mxfile) {
  const pages = [];
  const pattern =
    /<diagram\b([^>]*)>\s*(<mxGraphModel[\s\S]*?<\/mxGraphModel>)\s*<\/diagram>/g;
  let match;

  while ((match = pattern.exec(mxfile)) !== null) {
    const attrs = match[1];
    const nameMatch = attrs.match(/\bname="([^"]*)"/);
    pages.push({
      name: nameMatch ? decodeXmlEntities(nameMatch[1]) : `Page ${pages.length + 1}`,
      xml: match[2],
    });
  }

  if (pages.length === 0) {
    throw new Error(`No uncompressed draw.io pages found in ${INPUT}`);
  }

  return pages;
}

function parseDataUri(dataUri) {
  const comma = dataUri.indexOf(",");
  if (comma < 0) {
    throw new Error("Malformed SVG data URI");
  }

  const header = dataUri.slice(0, comma);
  const payload = dataUri.slice(comma + 1);
  return header.includes(";base64")
    ? Buffer.from(payload, "base64").toString("utf8")
    : decodeURIComponent(payload);
}

function svgDimensions(svg) {
  const root = svg.match(/<svg\b[^>]*>/i)?.[0] || "";
  const width = Number.parseFloat(root.match(/\bwidth="([\d.]+)/i)?.[1] || "");
  const height = Number.parseFloat(root.match(/\bheight="([\d.]+)/i)?.[1] || "");

  if (Number.isFinite(width) && Number.isFinite(height)) {
    return { width, height };
  }

  const viewBox = root
    .match(/\bviewBox="[^"]*?([\d.]+)\s+([\d.]+)"\s*$/i)?.slice(1)
    .map(Number);
  if (viewBox && viewBox.every(Number.isFinite)) {
    return { width: viewBox[0], height: viewBox[1] };
  }

  throw new Error("Could not determine SVG dimensions");
}

function normalizeEmbeddedImages(svg) {
  return svg.replace(
    /data:(image\/[^;"']+)%3Bbase64;base64,/gi,
    "data:$1;base64,",
  );
}

async function waitForEvent(page, event, timeout = 60000) {
  await page.waitForFunction(
    (wanted) => window.__drawioMessages.some((message) => message.event === wanted),
    event,
    { timeout },
  );
}

async function takeEvent(page, event) {
  return page.evaluate((wanted) => {
    const index = window.__drawioMessages.findIndex(
      (message) => message.event === wanted,
    );
    if (index < 0) return null;
    return window.__drawioMessages.splice(index, 1)[0];
  }, event);
}

async function renderSvgAssets(browser, svgInput, index, pageName) {
  const svg = normalizeEmbeddedImages(svgInput);
  const basename = OUTPUT_NAMES[index] || `figure${index + 1}`;
  const svgPath = path.join(OUTPUT, `${basename}.svg`);
  const pdfPath = path.join(OUTPUT, `${basename}.pdf`);
  const pngPath = path.join(OUTPUT, `${basename}.png`);
  const tiffPath = path.join(OUTPUT, `${basename}.tiff`);
  fs.writeFileSync(svgPath, svg, "utf8");

  const { width, height } = svgDimensions(svg);
  const context = await browser.newContext({
    viewport: {
      width: Math.ceil(width),
      height: Math.ceil(height),
    },
    // 2.4 x 1800 px gives a 4320 px preview, approximately 183 mm at
    // 600 dpi (AAAI/Nature-style double-column delivery width).
    deviceScaleFactor: 2.4,
  });
  const renderPage = await context.newPage();
  await renderPage.setContent(
    `<!doctype html><html><head><style>
      @page { size: ${width}px ${height}px; margin: 0; }
      html, body { width: ${width}px; height: ${height}px; margin: 0; padding: 0; overflow: hidden; background: #fff; }
      svg { display: block; width: ${width}px; height: ${height}px; }
    </style></head><body>${svg}</body></html>`,
    { waitUntil: "load" },
  );
  await renderPage.screenshot({ path: pngPath });
  await renderPage.pdf({
    path: pdfPath,
    width: `${width}px`,
    height: `${height}px`,
    printBackground: true,
    margin: { top: "0", right: "0", bottom: "0", left: "0" },
    pageRanges: "1",
  });
  await context.close();

  return {
    page: pageName,
    width,
    height,
    svg: path.relative(ROOT, svgPath),
    pdf: path.relative(ROOT, pdfPath),
    png: path.relative(ROOT, pngPath),
    tiff: path.relative(ROOT, tiffPath),
  };
}

async function exportPage(browser, entry, index) {
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  await page.setContent(
    `<!doctype html><html><head><style>
      html, body, iframe { width: 100%; height: 100%; margin: 0; border: 0; overflow: hidden; }
    </style></head><body>
      <script>
        window.__drawioMessages = [];
        window.addEventListener("message", (event) => {
          let message = event.data;
          if (typeof message === "string") {
            try { message = JSON.parse(message); } catch { return; }
          }
          if (message && typeof message === "object") {
            window.__drawioMessages.push(message);
          }
        });
      </script>
      <iframe id="drawio" src="https://embed.diagrams.net/?embed=1&proto=json&spin=1&libraries=0&ui=min"></iframe>
    </body></html>`,
    // app.min.js is roughly 9.5 MB. On the throttled research host it can
    // legitimately take more than one minute to arrive, so the old 60 s
    // navigation timeout failed just before Draw.io emitted `init`.
    { waitUntil: "load", timeout: DRAWIO_LOAD_TIMEOUT_MS },
  );
  await waitForEvent(page, "init", DRAWIO_LOAD_TIMEOUT_MS);
  await takeEvent(page, "init");

  await page.evaluate((xml) => {
    document.getElementById("drawio").contentWindow.postMessage(
      JSON.stringify({
        action: "load",
        xml,
        fit: 1,
        border: 0,
        background: "#ffffff",
        dark: false,
        noSaveBtn: 1,
        noExitBtn: 1,
      }),
      "*",
    );
  }, entry.xml);

  await waitForEvent(page, "load", 90000);
  await takeEvent(page, "load");

  await page.evaluate(() => {
    document.getElementById("drawio").contentWindow.postMessage(
      JSON.stringify({
        action: "export",
        format: "svg",
        border: 0,
        background: "#ffffff",
        theme: "light",
        embedImages: true,
        embedFonts: false,
        currentPage: true,
      }),
      "*",
    );
  });

  await waitForEvent(page, "export", 120000);
  const exported = await takeEvent(page, "export");
  if (!exported?.data) {
    throw new Error(`Draw.io returned no SVG for ${entry.name}`);
  }

  await context.close();
  return renderSvgAssets(browser, parseDataUri(exported.data), index, entry.name);
}

async function exportPageWithRetry(browser, entry, index, attempts = 3) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await exportPage(browser, entry, index);
    } catch (error) {
      lastError = error;
      for (const context of browser.contexts()) {
        await context.close().catch(() => {});
      }
      if (attempt < attempts) {
        process.stderr.write(
          `Draw.io export retry ${attempt}/${attempts} for ${entry.name}: ${error.message}\n`,
        );
        await new Promise((resolve) => setTimeout(resolve, 2000 * attempt));
      }
    }
  }

  throw lastError;
}

async function main() {
  fs.mkdirSync(OUTPUT, { recursive: true });
  const pages = extractPages(fs.readFileSync(INPUT, "utf8"));
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME,
    args: ["--disable-gpu", "--disable-dev-shm-usage"],
  });
  const manifest = [];

  try {
    if (process.argv.includes("--render-existing")) {
      for (let index = 0; index < pages.length; index += 1) {
        const svgPath = path.join(
          OUTPUT,
          `${OUTPUT_NAMES[index] || `figure${index + 1}`}.svg`,
        );
        manifest.push(
          await renderSvgAssets(
            browser,
            fs.readFileSync(svgPath, "utf8"),
            index,
            pages[index].name,
          ),
        );
      }
    } else {
      for (let index = 0; index < pages.length; index += 1) {
        manifest.push(await exportPageWithRetry(browser, pages[index], index));
      }
    }
  } finally {
    await browser.close();
  }

  execFileSync(process.env.PYTHON || "python", [TIFF_RENDERER], {
    cwd: ROOT,
    stdio: "inherit",
  });

  const manifestPath = path.join(OUTPUT, "drawio_export_manifest.json");
  fs.writeFileSync(
    manifestPath,
    `${JSON.stringify(
      {
        source: path.relative(ROOT, INPUT),
        exported_at: new Date().toISOString(),
        outputs: manifest,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
