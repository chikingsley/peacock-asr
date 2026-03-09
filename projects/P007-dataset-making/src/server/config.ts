import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "../..");
const dataRoot = path.resolve(projectRoot, "server", "data");
const referenceRoot = path.resolve(projectRoot, "references", "speechocean762");

export const config = {
  dataRoot,
  databasePath: path.join(dataRoot, "annotation.sqlite"),
  referenceRoot,
  resourcePath: path.join(referenceRoot, "resource"),
  waveRoot: path.join(referenceRoot, "WAVE"),
  splits: ["train", "test"] as const,
  defaultReviewerId: "demo-reviewer",
  defaultReviewerName: "Demo Reviewer",
};
