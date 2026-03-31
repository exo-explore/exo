/**
 * File attachment types for the chat interface
 */

import { getDocument, GlobalWorkerOptions, version } from "pdfjs-dist";
import type { DocumentInitParameters } from "pdfjs-dist/types/src/display/api";

GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${version}/build/pdf.worker.mjs`;

const PDF_PAGE_SCALE = 2.0;
const PDF_MAX_PAGES = 20;
const PDF_MAX_TEXT_CHARS = 100_000;

export interface ChatUploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
  preview?: string;
  textContent?: string;
  pageImages?: string[];
}

export interface ChatAttachment {
  type: "image" | "text" | "pdf" | "audio";
  name: string;
  content?: string;
  base64Url?: string;
  mimeType?: string;
}

export type FileCategory = "image" | "text" | "pdf" | "audio" | "unknown";

export const IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".webp",
  ".svg",
];
export const IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "image/svg+xml",
];

export const TEXT_EXTENSIONS = [
  ".txt",
  ".md",
  ".json",
  ".xml",
  ".yaml",
  ".yml",
  ".csv",
  ".log",
  ".js",
  ".ts",
  ".jsx",
  ".tsx",
  ".py",
  ".java",
  ".cpp",
  ".c",
  ".h",
  ".css",
  ".html",
  ".htm",
  ".sql",
  ".sh",
  ".bat",
  ".rs",
  ".go",
  ".rb",
  ".php",
  ".swift",
  ".kt",
  ".scala",
  ".r",
  ".dart",
  ".vue",
  ".svelte",
];
export const TEXT_MIME_TYPES = [
  "text/plain",
  "text/markdown",
  "text/csv",
  "text/html",
  "text/css",
  "application/json",
  "application/xml",
  "text/xml",
  "application/javascript",
  "text/javascript",
  "application/typescript",
];

export const PDF_EXTENSIONS = [".pdf"];
export const PDF_MIME_TYPES = ["application/pdf"];

export const AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a"];
export const AUDIO_MIME_TYPES = [
  "audio/mpeg",
  "audio/wav",
  "audio/ogg",
  "audio/mp4",
];

/**
 * Get file category based on MIME type and extension
 */
export function getFileCategory(
  mimeType: string,
  fileName: string,
): FileCategory {
  const extension = fileName.toLowerCase().slice(fileName.lastIndexOf("."));

  if (
    IMAGE_MIME_TYPES.includes(mimeType) ||
    IMAGE_EXTENSIONS.includes(extension)
  ) {
    return "image";
  }
  if (PDF_MIME_TYPES.includes(mimeType) || PDF_EXTENSIONS.includes(extension)) {
    return "pdf";
  }
  if (
    AUDIO_MIME_TYPES.includes(mimeType) ||
    AUDIO_EXTENSIONS.includes(extension)
  ) {
    return "audio";
  }
  if (
    TEXT_MIME_TYPES.includes(mimeType) ||
    TEXT_EXTENSIONS.includes(extension) ||
    mimeType.startsWith("text/")
  ) {
    return "text";
  }
  return "unknown";
}

/**
 * Get accept string for file input based on categories
 */
export function getAcceptString(categories: FileCategory[]): string {
  const accepts: string[] = [];

  for (const category of categories) {
    switch (category) {
      case "image":
        accepts.push(...IMAGE_EXTENSIONS, ...IMAGE_MIME_TYPES);
        break;
      case "text":
        accepts.push(...TEXT_EXTENSIONS, ...TEXT_MIME_TYPES);
        break;
      case "pdf":
        accepts.push(...PDF_EXTENSIONS, ...PDF_MIME_TYPES);
        break;
      case "audio":
        accepts.push(...AUDIO_EXTENSIONS, ...AUDIO_MIME_TYPES);
        break;
    }
  }

  return accepts.join(",");
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

/**
 * Read file as data URL (base64)
 */
export function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/**
 * Read file as text
 */
export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

async function extractPdfContent(
  file: File,
): Promise<{ text: string; pageImages: string[] }> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await getDocument({
    data: new Uint8Array(arrayBuffer),
    useSystemFonts: true,
  } as DocumentInitParameters).promise;

  const numPages = Math.min(pdf.numPages, PDF_MAX_PAGES);
  const pageTexts: string[] = [];
  const pageImages: string[] = [];

  for (let i = 1; i <= numPages; i++) {
    const page = await pdf.getPage(i);

    const content = await page.getTextContent();
    const strings = content.items
      .filter((item: any) => "str" in item)
      .map((item: any) => item.str as string);
    pageTexts.push(strings.join(" "));

    const viewport = page.getViewport({ scale: PDF_PAGE_SCALE });
    const canvas = new OffscreenCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext("2d");
    if (ctx) {
      await page.render({ canvasContext: ctx as any, viewport }).promise;
      const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.8 });
      const reader = new FileReader();
      const dataUrl = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      pageImages.push(dataUrl);
    }
  }

  let text = pageTexts.join("\n\n").trim();
  if (text.length > PDF_MAX_TEXT_CHARS) {
    text = text.slice(0, PDF_MAX_TEXT_CHARS) + "\n\n[truncated]";
  }
  if (pdf.numPages > PDF_MAX_PAGES) {
    text += `\n\n[showing ${PDF_MAX_PAGES} of ${pdf.numPages} pages]`;
  }

  return { text, pageImages };
}

/**
 * Process uploaded files into ChatUploadedFile format
 */
export async function processUploadedFiles(
  files: File[],
): Promise<ChatUploadedFile[]> {
  const results: ChatUploadedFile[] = [];

  for (const file of files) {
    const id =
      Date.now().toString() + Math.random().toString(36).substring(2, 9);
    const category = getFileCategory(file.type, file.name);

    const base: ChatUploadedFile = {
      id,
      name: file.name,
      size: file.size,
      type: file.type,
      file,
    };

    try {
      if (category === "image") {
        const preview = await readFileAsDataURL(file);
        results.push({ ...base, preview });
      } else if (category === "text" || category === "unknown") {
        const textContent = await readFileAsText(file);
        results.push({ ...base, textContent });
      } else if (category === "pdf") {
        const { text, pageImages } = await extractPdfContent(file);
        results.push({
          ...base,
          textContent: text || undefined,
          pageImages: pageImages.length > 0 ? pageImages : undefined,
        });
      } else if (category === "audio") {
        const preview = await readFileAsDataURL(file);
        results.push({ ...base, preview });
      } else {
        results.push(base);
      }
    } catch (error) {
      console.error("Error processing file:", file.name, error);
      results.push(base);
    }
  }

  return results;
}
