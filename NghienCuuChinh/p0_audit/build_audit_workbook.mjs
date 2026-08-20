import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve("p0_audit/outputs");
const audit = JSON.parse(await fs.readFile(path.join(root, "audit_report.json"), "utf8"));
const gpu = JSON.parse(await fs.readFile(path.join(root, "gpu_benchmark_512.json"), "utf8"));
const finalReport = JSON.parse(await fs.readFile(path.join(root, "P0_FINAL_REPORT.json"), "utf8"));
const trainCsv = await fs.readFile(path.join(root, "train_manifest.csv"), "utf8");
const validationCsv = await fs.readFile(path.join(root, "validation_manifest.csv"), "utf8");
const testCsv = await fs.readFile(path.join(root, "test_manifest_LOCKED_NO_AGE.csv"), "utf8");

const workbook = await Workbook.fromCSV(trainCsv, { sheetName: "Train Manifest" });
await workbook.fromCSV(validationCsv, { sheetName: "Validation Manifest" });
await workbook.fromCSV(testCsv, { sheetName: "Test Locked" });
const summary = workbook.worksheets.add("Tóm tắt P0");
const findings = workbook.worksheets.add("Kiểm tra QC");
const environment = workbook.worksheets.add("Môi trường");

const train = workbook.worksheets.getItem("Train Manifest");
const validation = workbook.worksheets.getItem("Validation Manifest");
const test = workbook.worksheets.getItem("Test Locked");

for (const sheet of [summary, train, validation, test, findings, environment]) {
  sheet.showGridLines = false;
}

summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["P0 – AUDIT DỮ LIỆU VÀ MÔI TRƯỜNG"]];
summary.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:F1").format.rowHeight = 30;

summary.getRange("A3:B3").values = [["Chỉ số", "Kết quả"]];
summary.getRange("A3:B3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
};
summary.getRange("A4:B15").values = [
  ["Train CSV", audit.counts.train_csv_rows],
  ["Train PNG", audit.counts.train_png],
  ["Test metadata", audit.counts.test_metadata_rows],
  ["Test PNG", audit.counts.test_png],
  ["Validation chính thức", finalReport.counts.validation],
  ["Ảnh train không đọc được", audit.image_integrity.unreadable_train.length],
  ["Ảnh test không đọc được", audit.image_integrity.unreadable_test.length],
  ["Nhóm hash trùng trong train", Object.keys(audit.image_integrity.duplicate_hash_groups_train).length],
  ["Nhóm hash trùng trong test", Object.keys(audit.image_integrity.duplicate_hash_groups_test).length],
  ["Hash trùng train–test", audit.image_integrity.cross_split_duplicate_hashes.length],
  ["Train manifest SHA-256", audit.fingerprints.train_manifest_sha256],
  ["Test locked manifest SHA-256", audit.fingerprints.test_locked_manifest_sha256],
];
summary.getRange("A4:A15").format.font = { bold: true, color: "#24364B" };
summary.getRange("B8").format = { fill: "#C6EFCE", font: { bold: true, color: "#006100" } };
summary.getRange("A17:F17").merge();
summary.getRange("A17").values = [["KẾT LUẬN TẠM THỜI"]];
summary.getRange("A17:F17").format = { fill: "#FFF2CC", font: { bold: true, color: "#7F6000" } };
summary.getRange("A18:F20").merge();
summary.getRange("A18").values = [[
  "P0 PASS: 12.611 train, 1.425 validation chính thức và 200 test đã đạt kiểm tra cấu trúc, " +
  "giải mã ảnh, ID, annotation và SHA-256. Validation khớp annotation Deeplasia. File nhãn test " +
  "được khóa; workbook này không chứa tuổi xương test."
]];
summary.getRange("A18:F20").format = { wrapText: true, verticalAlignment: "top", fill: "#FFFDF3" };
summary.getRange("A18:F20").format.borders = { preset: "outside", style: "thin", color: "#D6B656" };
summary.getRange("A22:B22").values = [["Benchmark 512 × 512", "Peak VRAM (MiB)"]];
summary.getRange("A22:B22").format = { fill: "#E2F0D9", font: { bold: true, color: "#375623" } };
const benchmarkRows = gpu.results.map((item) => [
  `${item.model}, batch ${item.batch_size}, ${item.status}`,
  item.peak_reserved_mib ?? null,
]);
summary.getRangeByIndexes(22, 0, benchmarkRows.length, 2).values = benchmarkRows;
summary.getRange(`B23:B${22 + benchmarkRows.length}`).format.numberFormat = "0.0";
summary.freezePanes.freezeRows(3);
summary.getRange("A1:F25").format.font = { name: "Aptos", size: 10 };
summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:B").format.columnWidth = 70;
summary.getRange("C:F").format.columnWidth = 12;

const checks = [
  ["Số dòng train CSV khớp PNG", audit.counts.train_csv_rows === audit.counts.train_png, 0, "12.611/12.611"],
  ["Số dòng test metadata khớp PNG", audit.counts.test_metadata_rows === audit.counts.test_png, 0, "200/200"],
  ["Không trùng ID train", audit.id_integrity.duplicate_train_csv_ids.length === 0, audit.id_integrity.duplicate_train_csv_ids.length, ""],
  ["Không thiếu ảnh train", audit.id_integrity.train_csv_without_image.length === 0, audit.id_integrity.train_csv_without_image.length, ""],
  ["Không có ảnh train thừa metadata", audit.id_integrity.train_image_without_csv.length === 0, audit.id_integrity.train_image_without_csv.length, ""],
  ["Không trùng ID train–test", audit.id_integrity.train_test_id_overlap.length === 0, audit.id_integrity.train_test_id_overlap.length, ""],
  ["Không ảnh hỏng", audit.image_integrity.unreadable_train.length + audit.image_integrity.unreadable_test.length === 0, audit.image_integrity.unreadable_train.length + audit.image_integrity.unreadable_test.length, ""],
  ["Không hash trùng train–test", audit.image_integrity.cross_split_duplicate_hashes.length === 0, audit.image_integrity.cross_split_duplicate_hashes.length, ""],
  ["Sex train hợp lệ", audit.label_integrity.invalid_train_sex_ids.length === 0, audit.label_integrity.invalid_train_sex_ids.length, ""],
  ["Tuổi train trong 0–228 tháng", audit.label_integrity.invalid_train_age_ids.length === 0, audit.label_integrity.invalid_train_age_ids.length, ""],
  ["ID file nhãn test khóa khớp metadata", audit.label_integrity.locked_test_ids_match_metadata, audit.label_integrity.locked_test_missing_ids.length + audit.label_integrity.locked_test_extra_ids.length, "Không xuất nhãn tuổi"],
  ["Có validation chính thức 1.425 ảnh", finalReport.counts.validation === 1425, 0, "PASS – khớp Deeplasia"],
];
findings.getRange("A1:D1").values = [["Kiểm tra", "Đạt", "Số lỗi/thiếu", "Ghi chú"]];
findings.getRangeByIndexes(1, 0, checks.length, 4).values = checks;
findings.getRange("A1:D1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
findings.getRange(`B2:B${checks.length + 1}`).format.numberFormat = "@";
findings.getRange(`B2:B${checks.length + 1}`).conditionalFormats.addCustom("=B2=TRUE", { fill: "#C6EFCE", font: { color: "#006100" } });
findings.getRange(`B2:B${checks.length + 1}`).conditionalFormats.addCustom("=B2=FALSE", { fill: "#FFC7CE", font: { color: "#9C0006" } });
findings.freezePanes.freezeRows(1);
findings.getRange("A:A").format.columnWidth = 42;
findings.getRange("B:B").format.columnWidth = 12;
findings.getRange("C:C").format.columnWidth = 16;
findings.getRange("D:D").format.columnWidth = 28;

const envRows = [
  ["Thuộc tính", "Giá trị"],
  ["Python", audit.environment.python],
  ["Pillow", audit.environment.pillow],
  ["Platform", gpu.environment.platform],
  ["PyTorch", gpu.environment.torch],
  ["timm", gpu.environment.timm],
  ["CUDA build", gpu.environment.cuda_build],
  ["cuDNN", gpu.environment.cudnn],
  ["GPU", gpu.environment.gpu],
  ["GPU total MiB", gpu.environment.gpu_total_mib],
  ["Train source", audit.source_root],
  ["Validation source", "D:\\Hoctap\\Doan_totnghiep\\Dataset\\RSNA\\boneage-validation-dataset"],
];
environment.getRangeByIndexes(0, 0, envRows.length, 2).values = envRows;
environment.getRange("A1:B1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
environment.getRange("A:A").format.columnWidth = 24;
environment.getRange("B:B").format.columnWidth = 90;
environment.getRange("B2:B12").format.wrapText = true;
environment.freezePanes.freezeRows(1);

for (const sheet of [train, validation, test]) {
  const used = sheet.getUsedRange();
  const header = used.getRow(0);
  header.format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  sheet.getRange("A:A").format.columnWidth = 15;
  sheet.getRange("B:B").format.columnWidth = 14;
  used.format.borders = { insideHorizontal: { style: "thin", color: "#E6EAF0" } };
}

train.getRange("C:C").format.columnWidth = 18;
train.getRange("D:D").format.columnWidth = 8;
train.getRange("E:E").format.columnWidth = 52;
train.getRange("F:F").format.columnWidth = 14;
train.getRange("G:G").format.columnWidth = 66;
train.getRange("H:I").format.columnWidth = 10;
train.getRange("J:K").format.columnWidth = 10;
train.getRange("L:L").format.columnWidth = 24;

validation.getRange("C:C").format.columnWidth = 18;
validation.getRange("D:D").format.columnWidth = 8;
validation.getRange("E:E").format.columnWidth = 52;
validation.getRange("F:F").format.columnWidth = 14;
validation.getRange("G:G").format.columnWidth = 66;
validation.getRange("H:I").format.columnWidth = 10;
validation.getRange("J:K").format.columnWidth = 10;
validation.getRange("L:L").format.columnWidth = 24;

test.getRange("C:C").format.columnWidth = 8;
test.getRange("D:D").format.columnWidth = 52;
test.getRange("E:E").format.columnWidth = 14;
test.getRange("F:F").format.columnWidth = 66;
test.getRange("G:H").format.columnWidth = 10;
test.getRange("I:J").format.columnWidth = 10;
test.getRange("K:K").format.columnWidth = 24;
environment.getRange("B2:B12").format.horizontalAlignment = "left";

const inspect = await workbook.inspect({
  kind: "table",
  range: "'Tóm tắt P0'!A1:F25",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 6,
  maxChars: 6000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Tóm tắt P0",
  range: "A1:F25",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(path.join(root, "P0_AUDIT_REPORT_preview.png"), new Uint8Array(await preview.arrayBuffer()));

const visualChecks = [
  ["Train Manifest", "A1:L15", "P0_preview_train_manifest.png"],
  ["Validation Manifest", "A1:L15", "P0_preview_validation_manifest.png"],
  ["Test Locked", "A1:K15", "P0_preview_test_locked.png"],
  ["Kiểm tra QC", "A1:D14", "P0_preview_qc.png"],
  ["Môi trường", "A1:B12", "P0_preview_environment.png"],
];
for (const [sheetName, range, fileName] of visualChecks) {
  const rendered = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(root, fileName), new Uint8Array(await rendered.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(root, "P0_AUDIT_REPORT.xlsx"));
