import { defineConfig } from "astro/config";

// Set REPO_NAME via env when building from Actions, or hardcode below.
const repo = process.env.REPO_NAME || "deadline-tracker";
const owner = process.env.REPO_OWNER || "yourname";

export default defineConfig({
  site: `https://${owner}.github.io`,
  base: `/${repo}`,
  outDir: "./dist",
});
