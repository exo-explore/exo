/**
 * File attachment types for the chat interface
 */

export interface ChatUploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
  preview?: string;
  textContent?: string;
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
        results.push(base);
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
