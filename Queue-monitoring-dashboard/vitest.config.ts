import { defineConfig } from "vitest/config";

/**
 * Vitest runs the pure TypeScript coordinate-transform + validation
 * module (`src/lib/mapping.ts`). Node environment: the module is DOM-free
 * by design, so the tests exercise the same production functions the
 * mapping editor renders with.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
