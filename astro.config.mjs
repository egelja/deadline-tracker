import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://deadlines.marusz.com",
  base: "/",
  outDir: "./dist",
  trailingSlash: "always",
});
