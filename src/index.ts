import TelegramBot, { TelegramApi } from '@codebam/cf-workers-telegram-bot';
import OpenAI from "openai";

//@ts-ignore
import { Buffer } from 'node:buffer';
import { isJPEGBase64 } from './isJpeg';
import { extractAllOGInfo } from "./og"

// Global set to track groups we've already notified the owner about (resets on deployment)
const notifiedGroups = new Set<number>();
function dispatchContent(content: string): { type: "text", text: string } | { type: "image_url", image_url: { url: string } } {
	if (content.startsWith("data:image/jpeg;base64,")) {
		return ({
			"type": "image_url",
			"image_url": {
				"url": content
			},
		})
	}
	return ({
		"type": "text",
		"text": content,
	});
}

function getMessageLink(r: { groupId: string, messageId: number }) {
	return `https://t.me/c/${parseInt(r.groupId.slice(2))}/${r.messageId}`;
}

function getSendTime(r: R) {
	return new Date(r.timeStamp).toLocaleString("en-US", { timeZone: "US/Eastern" });
}

function escapeMarkdownV2(text: string) {
	// Escape backslash first to avoid double-interpreting existing escapes.
	const withEscapedBackslashes = text.replace(/\\/g, "\\\\");
	const reservedChars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'];
	// Regex needs to escape special characters
	const escapedChars = reservedChars.map(char => '\\' + char).join('');
	const regex = new RegExp(`([${escapedChars}])`, 'g');
	return withEscapedBackslashes.replace(regex, '\\$1');
}

/**
 * Convert number to superscript number
 * @param {number} num - The number to convert
 * @returns {string} Number in superscript form
 */
export function toSuperscript(num: number) {
	const superscripts = {
		'0': '⁰',
		'1': '¹',
		'2': '²',
		'3': '³',
		'4': '⁴',
		'5': '⁵',
		'6': '⁶',
		'7': '⁷',
		'8': '⁸',
		'9': '⁹'
	};

	return num
		.toString()
		.split('')
		.map(digit => superscripts[digit as keyof typeof superscripts])
		.join('');
}
/**
 * Process duplicate links in Markdown text, converting them to sequential numbering format
 * @param {string} text - Input Markdown text
 * @param {Object} options - Configuration options
 * @param {string} options.prefix - Prefix for link text, defaults to "reference"
 * @param {boolean} options.useEnglish - Whether to use English (link1) instead of reference (reference1), defaults to false
 * @returns {string} Processed Markdown text
 */
export function processMarkdownLinks(text: string, options: { prefix: string, useEnglish: boolean } = {
	prefix: 'reference',
	useEnglish: false
}) {
	const {
		prefix,
		useEnglish
	} = options;

	// Used to store links that have already appeared
	const linkMap = new Map();
	let linkCounter = 1;

	// Regex to match markdown links
	const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g;

	return text.replace(linkPattern, (match, displayText, url) => {
		// Only process cases where display text and URL are identical
		if (displayText !== url) {
			return match; // keep as is
		}

		// If this URL has appeared before, use the existing number
		if (!linkMap.has(url)) {
			linkMap.set(url, linkCounter++);
		}
		const linkNumber = linkMap.get(url);

		// Decide whether to use Chinese or English format based on options
		const linkPrefix = useEnglish ? 'link' : prefix;

		// Return new format [reference1](originalURL) or [link1](originalURL)
		return `[${linkPrefix}${toSuperscript(linkNumber)}](${url})`;
	});
}

type R = {
	groupId: string;
	userName: string;
	content: string;
	messageId: number;
	timeStamp: number;
}

type Provider = "openai" | "anthropic" | "google" | "workers-ai";
type ModelConfig = {
	provider: Provider;
	model: string;
	label: string;
	noTemperature?: boolean;
};

// Provider -> AI Gateway compat-endpoint slug (workers-ai routes via the AI binding instead).
export const GATEWAY_PROVIDER_SLUG: Record<Exclude<Provider, "workers-ai">, string> = {
	openai: "openai",
	google: "google-ai-studio",
	anthropic: "anthropic",
};

const DEFAULT_MODEL_KEY = "gemini-3.5-flash";
const MODEL_REGISTRY: Record<string, ModelConfig> = {
	"gpt-5.5": { provider: "openai", model: "gpt-5.5", label: "GPT-5.5", noTemperature: true },
	"gpt-5.5-mini": { provider: "openai", model: "gpt-5.5-mini", label: "GPT-5.5 Mini", noTemperature: true },
	"gpt-5.4": { provider: "openai", model: "gpt-5.4", label: "GPT-5.4" },
	"gpt-5.4-nano": { provider: "openai", model: "gpt-5.4-nano", label: "GPT-5.4 Nano" },
	"gpt-5.4-mini": { provider: "openai", model: "gpt-5.4-mini", label: "GPT-5.4 Mini" },
	"gpt-4.1": { provider: "openai", model: "gpt-4.1", label: "GPT-4.1" },
	"gpt-4.1-mini": { provider: "openai", model: "gpt-4.1-mini", label: "GPT-4.1 Mini" },
	"gpt-4.1-nano": { provider: "openai", model: "gpt-4.1-nano", label: "GPT-4.1 Nano" },
	"o4-mini": { provider: "openai", model: "o4-mini", label: "o4-mini", noTemperature: true },
	"o3": { provider: "openai", model: "o3", label: "o3", noTemperature: true },
	"gemini-3.5-flash": { provider: "google", model: "gemini-3.5-flash", label: "Gemini 3.5 Flash" },
	"gemini-2.5-pro": { provider: "google", model: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
	"gemini-2.5-flash": { provider: "google", model: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
	"gemini-2.5-flash-lite": { provider: "google", model: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite" },
	"claude-3.7-sonnet": { provider: "anthropic", model: "claude-3-7-sonnet-latest", label: "Claude 3.7 Sonnet" },
	"claude-3.5-sonnet": { provider: "anthropic", model: "claude-3-5-sonnet-latest", label: "Claude 3.5 Sonnet" },
	"claude-3.5-haiku": { provider: "anthropic", model: "claude-3-5-haiku-latest", label: "Claude 3.5 Haiku" },
};

const DEFAULT_TEMPERATURE = 0.4;

let modelSettingsTableReady = false;
let botSettingsTableReady = false;

function normalizeModelKey(input: string) {
	return input.trim().toLowerCase();
}

export function getModelByKey(modelKey: string) {
	const normalized = normalizeModelKey(modelKey);
	const modelConfig = MODEL_REGISTRY[normalized];
	if (!modelConfig) {
		const customMatch = normalized.match(/^(openai|google|anthropic|workers-ai):(.+)$/);
		if (!customMatch) {
			return null;
		}
		const provider = customMatch[1] as Provider;
		const rawModel = customMatch[2].trim();
		if (!rawModel) {
			return null;
		}
		return {
			modelKey: `${provider}:${rawModel}`,
			modelConfig: {
				provider,
				model: rawModel,
				label: `${provider.toUpperCase()} custom (${rawModel})`,
			},
		};
	}
	return { modelKey: normalized, modelConfig };
}

function listModelKeys() {
	return Object.keys(MODEL_REGISTRY).sort();
}

function formatModelOptions() {
	const grouped = {
		openai: [] as string[],
		google: [] as string[],
		anthropic: [] as string[],
		"workers-ai": [] as string[],
	};
	for (const key of listModelKeys()) {
		const modelConfig = MODEL_REGISTRY[key];
		grouped[modelConfig.provider].push(`${key} (${modelConfig.label})`);
	}
	return [
		"OpenAI:",
		...grouped.openai,
		"",
		"Google:",
		...grouped.google,
		"",
		"Anthropic:",
		...grouped.anthropic,
		...(grouped["workers-ai"].length ? ["", "Workers AI:", ...grouped["workers-ai"]] : []),
	];
}

// Single OpenAI-SDK client pointed at the AI Gateway OpenAI-compat endpoint. Every
// external-provider call (openai/google/anthropic) goes through this — no more direct
// calls to api.openai.com / generativelanguage.googleapis.com / api.anthropic.com.
// Docs: https://developers.cloudflare.com/ai-gateway/usage/chat-completion/
function getGatewayClient(env: Env) {
	return new OpenAI({
		apiKey: env.WORKER_AI_TOKEN, // placeholder; the real per-provider key is sent per-request below
		baseURL: `https://gateway.ai.cloudflare.com/v1/${env.CF_ACCOUNT_ID}/${env.AI_GATEWAY_ID}/compat`,
		defaultHeaders: { "cf-aig-authorization": `Bearer ${env.WORKER_AI_TOKEN}` },
		timeout: 120_000,
	});
}

function getProviderApiKey(env: Env, provider: Exclude<Provider, "workers-ai">): string {
	const key = provider === "openai" ? env.OPENAI_API_KEY
		: provider === "google" ? env.GEMINI_API_KEY
		: env.ANTHROPIC_API_KEY;
	if (!key) {
		const envVar = provider === "openai" ? "OPENAI_API_KEY" : provider === "google" ? "GEMINI_API_KEY" : "ANTHROPIC_API_KEY";
		throw new Error(`${envVar} is not configured.`);
	}
	return key;
}

async function ensureModelSettingsTable(env: Env) {
	if (modelSettingsTableReady) {
		return;
	}
	await env.DB.prepare(`
		CREATE TABLE IF NOT EXISTS GroupModelSettings (
			groupId TEXT PRIMARY KEY,
			modelKey TEXT NOT NULL,
			updatedAt INTEGER NOT NULL,
			updatedBy INTEGER
		)
	`).run();
	modelSettingsTableReady = true;
}

async function getGroupModelSelection(env: Env, groupId: number) {
	await ensureModelSettingsTable(env);
	const selection = await env.DB.prepare(`
		SELECT modelKey
		FROM GroupModelSettings
		WHERE CAST(groupId AS INTEGER) = ?
		LIMIT 1
	`).bind(groupId).first<{ modelKey: string }>();

	if (!selection?.modelKey) {
		const fallback = getModelByKey(DEFAULT_MODEL_KEY)!;
		return fallback;
	}
	const selected = getModelByKey(selection.modelKey);
	if (!selected) {
		const fallback = getModelByKey(DEFAULT_MODEL_KEY)!;
		return fallback;
	}
	return selected;
}

async function setGroupModelSelection(env: Env, groupId: number, modelKey: string, updatedBy?: number) {
	await ensureModelSettingsTable(env);
	await env.DB.prepare(`
		INSERT OR REPLACE INTO GroupModelSettings(groupId, modelKey, updatedAt, updatedBy)
		VALUES (CAST(? AS INTEGER), ?, ?, ?)
	`).bind(groupId, modelKey, Date.now(), updatedBy ?? null).run();
}

// --- Image model registry (mirrors MODEL_REGISTRY / /model above) ---

type ImageProvider = "google" | "workers-ai";
type ImageModelConfig = {
	provider: ImageProvider;
	model: string;
	label: string;
};

const IMAGE_MODEL_OFF = "off";
const DEFAULT_IMAGE_MODEL_KEY = "nano-banana-2-lite";
const IMAGE_MODEL_REGISTRY: Record<string, ImageModelConfig> = {
	"nano-banana-2-lite": { provider: "google", model: "gemini-3.1-flash-lite-image", label: "Nano Banana 2 Lite" },
	"nano-banana-2": { provider: "google", model: "gemini-3.1-flash-image", label: "Nano Banana 2" },
	"flux-schnell": { provider: "workers-ai", model: "@cf/black-forest-labs/flux-1-schnell", label: "FLUX.1 [schnell]" },
};

type ImageModelSelection = { modelKey: string, modelConfig: ImageModelConfig | null };

export function getImageModelByKey(modelKey: string): ImageModelSelection | null {
	const normalized = normalizeModelKey(modelKey);
	if (normalized === IMAGE_MODEL_OFF) {
		return { modelKey: IMAGE_MODEL_OFF, modelConfig: null };
	}
	const modelConfig = IMAGE_MODEL_REGISTRY[normalized];
	if (!modelConfig) {
		const customMatch = normalized.match(/^(google|workers-ai):(.+)$/);
		if (!customMatch) {
			return null;
		}
		const provider = customMatch[1] as ImageProvider;
		const rawModel = customMatch[2].trim();
		if (!rawModel) {
			return null;
		}
		return {
			modelKey: `${provider}:${rawModel}`,
			modelConfig: { provider, model: rawModel, label: `${provider.toUpperCase()} custom (${rawModel})` },
		};
	}
	return { modelKey: normalized, modelConfig };
}

function listImageModelKeys() {
	return Object.keys(IMAGE_MODEL_REGISTRY).sort();
}

function formatImageModelOptions() {
	const grouped = { google: [] as string[], "workers-ai": [] as string[] };
	for (const key of listImageModelKeys()) {
		const modelConfig = IMAGE_MODEL_REGISTRY[key];
		grouped[modelConfig.provider].push(`${key} (${modelConfig.label})`);
	}
	return [
		"off (disable image generation)",
		"",
		"Google:",
		...grouped.google,
		"",
		"Workers AI:",
		...grouped["workers-ai"],
	];
}

let imageModelSettingsTableReady = false;

async function ensureImageModelSettingsTable(env: Env) {
	if (imageModelSettingsTableReady) {
		return;
	}
	await env.DB.prepare(`
		CREATE TABLE IF NOT EXISTS GroupImageModelSettings (
			groupId TEXT PRIMARY KEY,
			modelKey TEXT NOT NULL,
			updatedAt INTEGER NOT NULL,
			updatedBy INTEGER
		)
	`).run();
	imageModelSettingsTableReady = true;
}

async function getGroupImageModelSelection(env: Env, groupId: number): Promise<ImageModelSelection> {
	await ensureImageModelSettingsTable(env);
	const selection = await env.DB.prepare(`
		SELECT modelKey
		FROM GroupImageModelSettings
		WHERE CAST(groupId AS INTEGER) = ?
		LIMIT 1
	`).bind(groupId).first<{ modelKey: string }>();

	if (!selection?.modelKey) {
		return getImageModelByKey(DEFAULT_IMAGE_MODEL_KEY)!;
	}
	const selected = getImageModelByKey(selection.modelKey);
	if (!selected) {
		return getImageModelByKey(DEFAULT_IMAGE_MODEL_KEY)!;
	}
	return selected;
}

async function setGroupImageModelSelection(env: Env, groupId: number, modelKey: string, updatedBy?: number) {
	await ensureImageModelSettingsTable(env);
	await env.DB.prepare(`
		INSERT OR REPLACE INTO GroupImageModelSettings(groupId, modelKey, updatedAt, updatedBy)
		VALUES (CAST(? AS INTEGER), ?, ?, ?)
	`).bind(groupId, modelKey, Date.now(), updatedBy ?? null).run();
}

// Global key/value bot settings shared across every group (e.g. persona, temperature).
async function ensureBotSettingsTable(env: Env) {
	if (botSettingsTableReady) {
		return;
	}
	await env.DB.prepare(`
		CREATE TABLE IF NOT EXISTS BotSettings (
			key TEXT PRIMARY KEY,
			value TEXT,
			updatedAt INTEGER NOT NULL,
			updatedBy INTEGER
		)
	`).run();
	botSettingsTableReady = true;
}

async function getBotSetting(env: Env, key: string) {
	await ensureBotSettingsTable(env);
	const row = await env.DB.prepare(`
		SELECT value FROM BotSettings WHERE key = ? LIMIT 1
	`).bind(key).first<{ value: string }>();
	return row?.value ?? null;
}

async function setBotSetting(env: Env, key: string, value: string, updatedBy?: number) {
	await ensureBotSettingsTable(env);
	await env.DB.prepare(`
		INSERT OR REPLACE INTO BotSettings(key, value, updatedAt, updatedBy)
		VALUES (?, ?, ?, ?)
	`).bind(key, value, Date.now(), updatedBy ?? null).run();
}

async function deleteBotSetting(env: Env, key: string) {
	await ensureBotSettingsTable(env);
	await env.DB.prepare(`DELETE FROM BotSettings WHERE key = ?`).bind(key).run();
}

// --- Daily scheduled summaries (per-group send hour, US/Eastern) ---

// Skip a scheduled summary when the group had fewer messages than this in the
// last 24h — the old auto-summary feature was removed for sending empty summaries.
const MIN_SCHEDULED_SUMMARY_MESSAGES = 1;

let scheduleSettingsTableReady = false;

async function ensureScheduleSettingsTable(env: Env) {
	if (scheduleSettingsTableReady) {
		return;
	}
	await env.DB.prepare(`
		CREATE TABLE IF NOT EXISTS GroupScheduleSettings (
			groupId TEXT PRIMARY KEY,
			hourEt INTEGER NOT NULL,
			updatedAt INTEGER NOT NULL,
			updatedBy INTEGER
		)
	`).run();
	scheduleSettingsTableReady = true;
}

async function getGroupScheduleHour(env: Env, groupId: number): Promise<number | null> {
	await ensureScheduleSettingsTable(env);
	const row = await env.DB.prepare(`
		SELECT hourEt FROM GroupScheduleSettings WHERE CAST(groupId AS INTEGER) = ? LIMIT 1
	`).bind(groupId).first<{ hourEt: number }>();
	return row?.hourEt ?? null;
}

async function setGroupScheduleHour(env: Env, groupId: number, hourEt: number, updatedBy?: number) {
	await ensureScheduleSettingsTable(env);
	await env.DB.prepare(`
		INSERT OR REPLACE INTO GroupScheduleSettings(groupId, hourEt, updatedAt, updatedBy)
		VALUES (CAST(? AS INTEGER), ?, ?, ?)
	`).bind(groupId, hourEt, Date.now(), updatedBy ?? null).run();
}

async function deleteGroupSchedule(env: Env, groupId: number) {
	await ensureScheduleSettingsTable(env);
	await env.DB.prepare(`
		DELETE FROM GroupScheduleSettings WHERE CAST(groupId AS INTEGER) = ?
	`).bind(groupId).run();
}

/**
 * Parse a schedule time argument ("21", "21:00", "9:00") into an hour 0-23.
 * The cron fires hourly, so only top-of-the-hour times are accepted;
 * returns null for anything else.
 */
export function parseScheduleHour(input: string): number | null {
	const m = input.trim().match(/^(\d{1,2})(?::(\d{2}))?$/);
	if (!m) {
		return null;
	}
	const hour = parseInt(m[1]);
	const minute = m[2] ? parseInt(m[2]) : 0;
	if (hour > 23 || minute !== 0) {
		return null;
	}
	return hour;
}

function formatScheduleHour(hourEt: number) {
	return `${hourEt.toString().padStart(2, "0")}:00 US/Eastern`;
}

// Resolve the personality override (universal across all groups) and temperature.
async function getBotPersona(env: Env) {
	return await getBotSetting(env, "persona");
}

async function getBotTemperature(env: Env) {
	const raw = await getBotSetting(env, "temperature");
	const parsed = raw == null ? NaN : parseFloat(raw);
	return Number.isFinite(parsed) ? parsed : DEFAULT_TEMPERATURE;
}

type DispatchContent = ReturnType<typeof dispatchContent>;
type ChatMessage = {
	role: "system" | "user" | "assistant";
	content: string | DispatchContent[];
};

// @cf/* chat models take plain-text message content; flatten multimodal blocks to text
// (image parts are dropped — only vision-capable Workers AI models handle images, and
// none of the current registry entries are vision models).
function flattenToText(content: string | DispatchContent[]): string {
	if (typeof content === "string") {
		return content;
	}
	return content.filter((b): b is { type: "text", text: string } => b.type === "text").map((b) => b.text).join("\n");
}

async function createModelResponse(
	env: Env,
	selectedModel: { modelKey: string, modelConfig: ModelConfig },
	messages: ChatMessage[],
	maxTokens = 4096,
	temperature = DEFAULT_TEMPERATURE,
) {
	const { provider, model, noTemperature } = selectedModel.modelConfig;

	if (provider === "workers-ai") {
		const response: any = await env.AI.run(
			model as any,
			{
				messages: messages.map((m) => ({ role: m.role, content: flattenToText(m.content) })),
				max_tokens: maxTokens,
				...(noTemperature ? {} : { temperature }),
			} as any,
			{ gateway: { id: env.AI_GATEWAY_ID } },
		);
		return response?.response || "";
	}

	if (!env.WORKER_AI_TOKEN) {
		throw new Error("WORKER_AI_TOKEN is not configured.");
	}
	const apiKey = getProviderApiKey(env, provider);
	const client = getGatewayClient(env);
	const response = await client.chat.completions.create(
		{
			model: `${GATEWAY_PROVIDER_SLUG[provider]}/${model}`,
			messages: messages as any,
			max_completion_tokens: maxTokens,
			...(noTemperature ? {} : { temperature }),
		},
		{ headers: { Authorization: `Bearer ${apiKey}` } },
	);
	return response.choices[0].message.content || "";
}

// Notify owner about non-whitelisted group (only once per deployment)
async function notifyOwnerAboutGroup(bot: TelegramApi, env: Env, groupId: number, groupName: string) {
	if (notifiedGroups.has(groupId)) {
		return; // Already notified
	}

	try {
		const ownerUserId = parseInt(env.OWNER_ID);
		await (bot as any).sendMessage(ownerUserId,
			`Bot received message from non-whitelisted group: ${groupName} (ID: ${groupId})\n\nUse /whitelist ${groupId} to approve this group for processing.`
		);
		notifiedGroups.add(groupId);
		console.debug(`Owner notified about group ${groupId}`);
	} catch (e) {
		console.error('Failed to notify owner about non-whitelisted group:', e);
	}
}

// Default personality/persona lines. The bot admin can override these on the fly
// with /persona; a single override applies universally to both summarize and ask.
const DEFAULT_PERSONA_SUMMARIZE = `You are a professional group chat summarization assistant. Your task is to summarize conversations in a natural, group-chat-friendly tone, in English only.`;
const DEFAULT_PERSONA_ANSWER = `You are an intelligent group chat assistant. Your task is to answer user questions based on the provided chat history, in English only.`;

// The non-personality guidance for each scenario. These stay fixed regardless of persona.
const PROMPT_BODY_SUMMARIZE = `

The conversation will be provided in the following format:
====================
Username:
Message content
Associated link
====================

Follow these guidelines:
1. If multiple topics are discussed, summarize them as separate sections, clearly indicating topic shifts. Give each topic its own heading (e.g. "Topic 1: ...", "Topic 2: ...").
2. If images are mentioned, include relevant descriptions in the summary.
3. Reference original messages with inline links: [keyword](URL). Prefer a meaningful keyword as the link text; if none fits, use Ref + number. ALWAYS include the actual URL in parentheses — never output a bare "[Ref]" or "[link]" without its URL.
4. Keep the summary concise while capturing key content and sentiment.
5. Start the summary with the one-line time frame and message count information provided in the request.
6. Output must be entirely in English, but it is fine to quote non-English content from the chat as long as the summary itself is in English.
7. For each topic add a brief bullet paragraph offset note labeled "AI context:" — 1-2 factual, useful pieces of information relevant to the topic (background facts, current data, clarifications, or counterpoints from general knowledge). Do NOT comment on the tone, humor, or sentiment of the conversation.
8. Keep the total response within ~256 words and be as concise as possible.

Formatting: respond in clean GitHub-Flavored Markdown. Use formatting freely where it improves readability — text color, headings, **bold**, _italic_, ~~strikethrough~~, bullet and numbered lists (nesting allowed), > blockquotes, tables, fenced code blocks, and inline [text](url) links, and markdown color. Use LaTeX for any math ($x^2$ inline, $$...$$ for block). Do not wrap the whole response in a code block.`;

const PROMPT_BODY_ANSWER = `

The chat history will be provided in the following format:
====================
Username:
Message content
Associated link
====================

Follow these guidelines:
1. Answer the question directly and concisely based on the chat history.
2. If images are mentioned, include relevant descriptions where relevant.
3. Reference original messages with inline links: [keyword](URL). Prefer a meaningful keyword as the link text; if none fits, use Ref + number. ALWAYS include the actual URL in parentheses — never output a bare "[Ref]" or "[link]" without its URL.
4. Output must be entirely in English, but it is fine to quote non-English content from the chat as long as the answer itself is in English.
5. Keep the answer concise (within ~256 words) unless the question genuinely requires more detail.

Formatting: respond in clean GitHub-Flavored Markdown. Use formatting freely where it improves readability — text color, headings, **bold**, _italic_, ~~strikethrough~~, bullet and numbered lists (nesting allowed), > blockquotes, tables, fenced code blocks, and inline [text](url) links. Use LaTeX for any math ($x^2$ inline, $$...$$ for block). Do not wrap the whole response in a code block.`;

// Build the system prompts, applying the admin persona override (if any) to both
// scenarios. When no override is set, each scenario keeps its own default persona.
function buildSystemPrompts(personaOverride: string | null) {
	const summarizePersona = personaOverride || DEFAULT_PERSONA_SUMMARIZE;
	const answerPersona = personaOverride || DEFAULT_PERSONA_ANSWER;
	return {
		summarizeChat: `${summarizePersona}${PROMPT_BODY_SUMMARIZE}`,
		answerQuestion: `${answerPersona}${PROMPT_BODY_ANSWER}`,
	};
}

function getCommandVar(str: string, delim: string) {
	return str.slice(str.indexOf(delim) + delim.length);
}

type ImageBytes = { bytes: Uint8Array; mime: string };

/**
 * Send a message as Telegram rich content (Bot API 10.1 sendRichMessage).
 * The model's GitHub-Flavored Markdown is passed verbatim in rich_message.markdown
 * and parsed into rich blocks server-side — no MarkdownV2 escaping required.
 * To attach an image, embed a real HTTPS URL via standard markdown image syntax
 * (`![...](url)`) in `markdown` before calling this — no file upload needed here.
 * Returns { ok } so callers can show a notice / fall back on failure.
 */
async function sendRichMessage(
	env: Env,
	chatId: number | string,
	markdown: string,
	replyToMessageId?: number,
): Promise<{ ok: boolean; description?: string }> {
	const body: Record<string, unknown> = {
		chat_id: chatId,
		rich_message: { markdown },
	};
	if (replyToMessageId !== undefined) {
		body.reply_parameters = { message_id: replyToMessageId };
	}
	const res = await fetch(`https://api.telegram.org/bot${env.SECRET_TELEGRAM_API_TOKEN}/sendRichMessage`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
	const data = await res.json<any>().catch(() => null);
	if (!data?.ok) {
		console.error("sendRichMessage failed", res.status, JSON.stringify(data?.description));
		return { ok: false, description: data?.description };
	}
	return { ok: true };
}

/**
 * Stream an ephemeral (~30s) draft preview via sendRichMessageDraft while a summary
 * is being generated. Best-effort only: per Bot API docs chat_id here is scoped to
 * "the target private chat", so this may simply no-op/fail in group chats — that's
 * fine, it's just a "generating…" hint and must never block the real sendRichMessage.
 */
async function sendRichMessageDraft(
	env: Env,
	chatId: number | string,
	draftId: number,
	markdown: string,
): Promise<void> {
	try {
		const res = await fetch(`https://api.telegram.org/bot${env.SECRET_TELEGRAM_API_TOKEN}/sendRichMessageDraft`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ chat_id: chatId, draft_id: draftId, rich_message: { markdown } }),
		});
		const data = await res.json<any>().catch(() => null);
		if (!data?.ok) {
			console.debug("sendRichMessageDraft skipped/failed", res.status, JSON.stringify(data?.description));
		}
	} catch (e) {
		console.debug("sendRichMessageDraft errored", e);
	}
}

/** Fallback for when sendRichMessage's image markup is rejected: plain sendPhoto by URL, no caption. */
async function sendPhoto(
	env: Env,
	chatId: number | string,
	photoUrl: string,
	replyToMessageId?: number,
): Promise<{ ok: boolean; description?: string }> {
	const body: Record<string, unknown> = {
		chat_id: chatId,
		photo: photoUrl,
	};
	if (replyToMessageId !== undefined) {
		body.reply_parameters = { message_id: replyToMessageId };
	}
	const res = await fetch(`https://api.telegram.org/bot${env.SECRET_TELEGRAM_API_TOKEN}/sendPhoto`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
	const data = await res.json<any>().catch(() => null);
	if (!data?.ok) {
		console.error("sendPhoto failed", res.status, JSON.stringify(data?.description));
		return { ok: false, description: data?.description };
	}
	return { ok: true };
}

const R2_KEY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/** Random 14-char alphanumeric string, used as an unguessable R2 object filename. */
export function randomAlphanumeric(length = 14): string {
	const bytes = new Uint8Array(length);
	crypto.getRandomValues(bytes);
	return Array.from(bytes, (b) => R2_KEY_ALPHABET[b % R2_KEY_ALPHABET.length]).join("");
}

const MIME_EXTENSIONS: Record<string, string> = {
	"image/png": "png",
	"image/jpeg": "jpg",
	"image/webp": "webp",
	"image/gif": "gif",
};

/**
 * Upload a generated summary image to R2 and return its public (r2.dev) URL.
 * Keyed by group so images are easy to find/purge per group; filename is a random
 * 14-char string so URLs can't be guessed/enumerated. Returns null on any failure.
 */
async function uploadSummaryImageToR2(env: Env, groupId: number, image: ImageBytes): Promise<string | null> {
	try {
		const ext = MIME_EXTENSIONS[image.mime] || "png";
		const key = `${groupId}/${randomAlphanumeric(14)}.${ext}`;
		await env.SUMMARY_IMAGES.put(key, image.bytes, { httpMetadata: { contentType: image.mime } });
		return `${env.R2_PUBLIC_URL}/${key}`;
	} catch (e) {
		console.error("uploadSummaryImageToR2 failed", e);
		return null;
	}
}

/** Pull the first inline image (base64 + mime type) out of a Gemini generateContent response. */
export function extractInlineImage(data: any): { mimeType: string; data: string } | null {
	const parts = data?.candidates?.[0]?.content?.parts || [];
	const imagePart = parts.find((p: any) => p?.inlineData?.data);
	return imagePart ? imagePart.inlineData : null;
}

/**
 * Generate an illustration for a /summary via the group's chosen image model.
 * Never throws — returns null (and logs) on any failure so an image outage
 * can't block the summary itself from being sent.
 * If the persona-flavored prompt fails (e.g. the provider's content-safety
 * filter rejects it), a second attempt is made without the persona text.
 */
async function generateSummaryImage(
	env: Env,
	imageSelection: { modelKey: string, modelConfig: ImageModelConfig },
	summaryText: string,
	persona: string | null = null,
): Promise<ImageBytes | null> {
	const attempt = async (personaText: string | null): Promise<ImageBytes> => {
		const prompt = personaText
			? `${personaText}\n\ncreate an image for the following summary\n\n${summaryText}`
			: `create an image for the following summary\n\n${summaryText}`;
		if (imageSelection.modelConfig.provider === "google") {
			if (!env.GEMINI_API_KEY) {
				throw new Error("GEMINI_API_KEY is not configured.");
			}
			const res = await fetch(
				`https://gateway.ai.cloudflare.com/v1/${env.CF_ACCOUNT_ID}/${env.AI_GATEWAY_ID}/google-ai-studio/v1beta/models/${imageSelection.modelConfig.model}:generateContent`,
				{
					method: "POST",
					headers: {
						"content-type": "application/json",
						"cf-aig-authorization": `Bearer ${env.WORKER_AI_TOKEN}`,
						"x-goog-api-key": env.GEMINI_API_KEY,
					},
					body: JSON.stringify({
						contents: [{ role: "user", parts: [{ text: prompt }] }],
						generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
					}),
				},
			);
			if (!res.ok) {
				throw new Error(`Image generation failed: ${res.status} ${await res.text()}`);
			}
			const data = await res.json<any>();
			const inlineImage = extractInlineImage(data);
			if (!inlineImage) {
				// Safety blocks come back as 200 with no image part; surface the reason for the log.
				const finishReason = data?.candidates?.[0]?.finishReason;
				const blockReason = data?.promptFeedback?.blockReason;
				throw new Error(`No inline image data in generateContent response (finishReason: ${finishReason ?? "n/a"}, blockReason: ${blockReason ?? "n/a"})`);
			}
			return {
				bytes: Uint8Array.from(Buffer.from(inlineImage.data, "base64")),
				mime: inlineImage.mimeType || "image/png",
			};
		}

		// workers-ai
		const response: any = await env.AI.run(
			imageSelection.modelConfig.model as any,
			{ prompt } as any,
			{ gateway: { id: env.AI_GATEWAY_ID } },
		);
		if (!response?.image) {
			throw new Error("No image data in Workers AI response");
		}
		return { bytes: Uint8Array.from(Buffer.from(response.image, "base64")), mime: "image/jpeg" };
	};

	try {
		return await attempt(persona);
	} catch (e) {
		if (!persona) {
			console.error("generateSummaryImage failed", e);
			return null;
		}
		console.error("generateSummaryImage failed with persona, retrying without persona", e);
		try {
			return await attempt(null);
		} catch (e2) {
			console.error("generateSummaryImage retry without persona failed", e2);
			return null;
		}
	}
}

/**
 * Full summary pipeline shared by the /summary command and the daily scheduled
 * summaries: model call with the group's model/persona/temperature, optional AI
 * illustration (uploaded to R2), and rich-message delivery with photo fallback.
 * Throws if the model call fails; returns { ok: false } if delivery failed.
 */
async function generateAndSendSummary(
	env: Env,
	groupId: number,
	results: Record<string, unknown>[],
	summaryHeader: string,
	replyToMessageId?: number,
): Promise<{ ok: boolean }> {
	const selectedModel = await getGroupModelSelection(env, groupId);
	const persona = await getBotPersona(env);
	const systemPrompts = buildSystemPrompts(persona);
	const botTemperature = await getBotTemperature(env);

	const rawSummary = await createModelResponse(
		env,
		selectedModel,
		[
			{
				role: "system",
				content: systemPrompts.summarizeChat,
			},
			{
				role: "user",
				content: [
					dispatchContent(`Please summarize this chat history.\n${summaryHeader}`),
					...results.flatMap(
						(r: any) => [
							dispatchContent(`====================`),
							dispatchContent(`${r.userName}:`),
							dispatchContent(r.content),
							dispatchContent(getMessageLink(r)),
						]
					),
				],
			},
		],
		4096,
		botTemperature,
	);

	// Pass the model's GitHub-Flavored Markdown verbatim to Telegram as a rich message.
	const plainSummaryMarkdown = `**Summary by ${selectedModel.modelKey}**\n\n${fixLink(rawSummary)}`;

	const imageSelection = await getGroupImageModelSelection(env, groupId);
	const imageModelConfig = imageSelection.modelConfig;
	const imageBytes = imageModelConfig
		? await generateSummaryImage(env, { modelKey: imageSelection.modelKey, modelConfig: imageModelConfig }, plainSummaryMarkdown, persona)
		: null;
	// Uploaded to our own R2 bucket (group-scoped, unguessable filename) so it can be
	// embedded as a plain HTTPS URL — attach://-style multipart uploads to sendRichMessage
	// are not actually supported by the Bot API.
	const imageUrl = imageBytes ? await uploadSummaryImageToR2(env, groupId, imageBytes) : null;

	const summaryMarkdown = imageUrl
		? `**Summary by ${selectedModel.modelKey} · image by ${imageSelection.modelKey}**\n\n${fixLink(rawSummary)}\n\n![Summary illustration](${imageUrl})`
		: plainSummaryMarkdown;

	const sent = await sendRichMessage(env, groupId, summaryMarkdown, replyToMessageId);

	if (!sent.ok) {
		if (imageUrl) {
			// Image markup rejected by the rich-message API — fall back to a
			// plain photo reply followed by the text-only rich summary.
			await sendPhoto(env, groupId, imageUrl, replyToMessageId);
			const textOnlySent = await sendRichMessage(env, groupId, plainSummaryMarkdown, replyToMessageId);
			return { ok: textOnlySent.ok };
		}
		return { ok: false };
	}
	return { ok: true };
}

/**
 *
 * @param text
 * @description I dont know why, but llm keep output tme.cat, so we need to fix it
 * @returns
 */
function fixLink(text: string) {
	return text.replace(/tme\.cat/g, "t.me/c").replace(/\/c\/c/g, "/c");
}

function getUserName(msg: any) {
	if (msg?.sender_chat?.title) {
		return msg.sender_chat.title as string;
	}
	return msg.from?.first_name as string || "anonymous";
}
export default {
	async scheduled(
		controller: ScheduledController,
		env: Env,
		ctx: ExecutionContext,
	) {
		console.debug("Scheduled task starting:", new Date().toISOString());
		const date = new Date(new Date().toLocaleString("en-US", { timeZone: "US/Eastern" }));

		// Send daily summaries to groups scheduled for the current hour (US/Eastern).
		// Only whitelisted groups qualify, and quiet groups are skipped so we never
		// send an empty summary.
		await ensureScheduleSettingsTable(env);
		const { results: dueGroups } = await env.DB.prepare(`
			SELECT s.groupId
			FROM GroupScheduleSettings s
			JOIN WhitelistedGroups w ON CAST(w.groupId AS INTEGER) = CAST(s.groupId AS INTEGER)
			WHERE s.hourEt = ?
		`).bind(date.getHours()).all<{ groupId: string }>();
		for (const dueGroup of dueGroups ?? []) {
			const groupId = parseInt(dueGroup.groupId);
			try {
				const { results } = await env.DB.prepare(`
					SELECT * FROM Messages
					WHERE groupId=? AND timeStamp >= ?
					ORDER BY timeStamp ASC
					`)
					.bind(groupId, Date.now() - 24 * 60 * 60 * 1000)
					.all();
				if (!results || results.length < MIN_SCHEDULED_SUMMARY_MESSAGES) {
					console.debug(`Skipping scheduled summary for group ${groupId}: only ${results?.length ?? 0} messages in the last 24h`);
					continue;
				}
				const firstMessageTime = getSendTime(results[0] as R);
				const lastMessageTime = getSendTime(results[results.length - 1] as R);
				const summaryHeader = `Daily chat summary for the last 24 hours\nTime frame: ${firstMessageTime} to ${lastMessageTime}\nMessage count: ${results.length}`;
				await generateAndSendSummary(env, groupId, results, summaryHeader);
				console.debug(`Scheduled summary sent to group ${groupId}`);
			} catch (e) {
				console.error(`Scheduled summary failed for group ${groupId}`, e);
			}
		}

		// Clean up oldest 4000 messages
		if (date.getHours() === 0 && date.getMinutes() < 5) {
			await env.DB.prepare(`
					DELETE FROM Messages
					WHERE id IN (
						SELECT id
						FROM (
							SELECT
								id,
								ROW_NUMBER() OVER (
									PARTITION BY groupId
									ORDER BY timeStamp DESC
								) as row_num
							FROM Messages
						) ranked
						WHERE row_num > 3000
					);`)
				.run();
		}
		// clean up old images
		if (date.getHours() === 0 && date.getMinutes() < 5) {
			ctx.waitUntil(env.DB.prepare(`
					DELETE
					FROM Messages
					WHERE timeStamp < ? AND content LIKE 'data:image/jpeg;base64,%'`)
				.bind(Date.now() - 24 * 60 * 60 * 1000)
				.run());
		}
		console.debug("cron processed");
	},
	fetch: async (request: Request, env: Env, ctx: ExecutionContext) => {
		await new TelegramBot(env.SECRET_TELEGRAM_API_TOKEN)
			.on('status', async (ctx) => {
				const res = (await ctx.reply('My house is quite big'))!;
				if (!res.ok) {
					console.error(`Error sending message:`, res);
				}
				return new Response('ok');
			})
			.on("query", async (ctx) => {
			const groupId = ctx.update.message!.chat.id; // numeric ID
				// Check whitelist
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
					await ctx.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					// Notify owner about this non-whitelisted group
					const groupName = ctx.update.message!.chat.title || 'Unknown';
					await notifyOwnerAboutGroup(ctx.api, env, groupId, groupName);
					return new Response('ok');
				}

				const messageText = ctx.update.message!.text || "";
				if (!messageText.split(" ")[1]) {
					const res = (await ctx.reply('Please enter the keyword to search'))!;
					if (!res.ok) {
						console.error(`Error sending message:`, res);
					}
					return new Response('ok');
				}
				const { results } = await env.DB.prepare(`
					SELECT * FROM Messages
					WHERE groupId=? AND content GLOB ?
					ORDER BY timeStamp DESC
					LIMIT 2000`)
					.bind(groupId, `*${messageText.split(" ")[1]}*`)
					.all();
				const res = (await ctx.reply(
					escapeMarkdownV2(`Search results:
${results.map((r: any) => `${r.userName}: ${r.content} ${r.messageId == null ? "" : `[link](https://t.me/c/${parseInt(r.groupId.slice(2))}/${r.messageId})`}`).join('\n')}`), "MarkdownV2"))!;
				if (!res.ok) {
					console.error(`Error sending message:`, res.status, res.statusText, await res.text());
				}
				return new Response('ok');
			})
			.on("ask", async (ctx) => {
			const groupId = ctx.update.message!.chat.id; // numeric ID
			const userId = ctx.update.message?.from?.id;
			const repliedMessage = (ctx.update.message as any)?.reply_to_message;
				
				// Check whitelist
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
				await ctx.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					const groupName = ctx.update.message!.chat.title || 'Unknown';
					await notifyOwnerAboutGroup(ctx.api, env, groupId, groupName);
					return new Response('ok');
				}

				const messageText = ctx.update.message!.text || "";
				if (!messageText.split(" ")[1]) {
					const res = (await ctx.reply('Please enter the question to ask'))!;
					if (!res.ok) {
						console.error(`Error sending message:`, res);
					}
					return new Response('ok');
				}
				if (!userId) {
					await ctx.reply(`Unable to identify requesting user.`);
					return new Response('ok');
				}
				const selectedModel = await getGroupModelSelection(env, groupId);
				let res = await ctx.api.sendMessage(ctx.bot.api.toString(), {
					"chat_id": userId,
					"parse_mode": "",
					"text": `Bot has received your question, please wait. Using model: ${selectedModel.modelKey}`,
					reply_to_message_id: -1,
				});
				if (!res.ok) {
					await ctx.reply(`Please start a private chat with the bot, otherwise unable to receive messages`);
					return new Response('ok');
				}
				const question = getCommandVar(messageText, " ");
				let modelUserContent: any[] = [];
				if (repliedMessage?.message_id) {
					const repliedContent = repliedMessage.text || repliedMessage.caption || "[non-text message]";
					const repliedUser = getUserName(repliedMessage);
					const repliedLink = getMessageLink({ groupId: groupId.toString(), messageId: repliedMessage.message_id });
					modelUserContent = [
						dispatchContent(`The user asked /ask by replying to a specific message. Focus on the replied message and do not summarize the entire chat unless explicitly requested.`),
						dispatchContent(`====================`),
						dispatchContent(`${repliedUser}:`),
						dispatchContent(repliedContent),
						dispatchContent(repliedLink),
						dispatchContent(`====================`),
					];
				} else {
					const { results } = await env.DB.prepare(`
						WITH latest_1000 AS (
							SELECT * FROM Messages
							WHERE groupId=?
							ORDER BY timeStamp DESC
							LIMIT 1000
						)
						SELECT * FROM latest_1000
						ORDER BY timeStamp ASC
						`)
						.bind(groupId)
						.all();
					modelUserContent = results.flatMap(
						(r: any) => [
							dispatchContent(`====================`),
							dispatchContent(`${r.userName}:`),
							dispatchContent(r.content),
							dispatchContent(getMessageLink(r)),
						]
					);
				}
				let answerText = "";
				try {
					const persona = await getBotPersona(env);
					const systemPrompts = buildSystemPrompts(persona);
					const botTemperature = await getBotTemperature(env);
					answerText = await createModelResponse(
						env,
						selectedModel,
						[
							{
								role: "system",
								content: systemPrompts.answerQuestion,
							},
							{
								role: "user",
								content: modelUserContent,
							},
							{
								role: "user",
								content: repliedMessage?.message_id
									? `Question about the replied message: ${question}`
									: `Question: ${question}`,
							},
						],
						4096,
						botTemperature,
					);
				} catch (e) {
					console.error(e);
					await ctx.reply(`Model call failed: ${(e as Error).message}`);
					return new Response('ok');
				}
				// Send the model's GitHub-Flavored Markdown verbatim as a rich message.
				const sent = await sendRichMessage(env, userId, fixLink(answerText || ""));
				if (!sent.ok) {
					await ctx.reply(`⚠️ Unable to generate enhanced markdown.`);
				}
				return new Response('ok');
			})
			.on("summary", async (bot) => {
			const groupId = bot.update.message!.chat.id; // numeric ID
				// Check whitelist
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				console.debug(`Summary check - groupId: ${groupId}, whitelistResults:`, whitelistResults);

				if (!whitelistResults || whitelistResults.length === 0) {
					await bot.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					// Notify owner about this non-whitelisted group
					const groupName = bot.update.message!.chat.title || 'Unknown';
					await notifyOwnerAboutGroup(bot.api, env, groupId, groupName);
					return new Response('ok');
				}

				if (bot.update.message!.text!.split(" ").length === 1) {
					await bot.reply('Please enter the time range/number of messages to query, e.g. /summary 114h or /summary 514');
					return new Response('ok');
				}
				const summary = bot.update.message!.text!.split(" ")[1];
				let results: Record<string, unknown>[];
				try {
					const test = parseInt(summary);
					if (Number.isNaN(test)) {
						throw new Error("not a number");
					}
					if (test < 0) {
						throw new Error("negative number");
					}
					if (!Number.isFinite(test)) {
						throw new Error("infinite number");
					}
				}
				catch (e: any) {
					await bot.reply('Please enter the time range/number of messages to query, e.g. /summary 114h or /summary 514  ' + e.message);
					return new Response('ok');
				}
				if (summary.endsWith("h")) {
					results = (await env.DB.prepare(`
						SELECT *
						FROM Messages
						WHERE groupId=? AND timeStamp >= ?
						ORDER BY timeStamp ASC
						`)
						.bind(groupId, Date.now() - parseInt(summary) * 60 * 60 * 1000)
						.all()).results;
				}
				else {
					results = (await env.DB.prepare(`
						WITH latest_n AS (
							SELECT * FROM Messages
							WHERE groupId=?
							ORDER BY timeStamp DESC
							LIMIT ?
						)
						SELECT * FROM latest_n
						ORDER BY timeStamp ASC
						`)
						.bind(groupId, Math.min(parseInt(summary), 4000))
						.all()).results;
				}
				if (results.length > 0) {
					try {
						// Calculate actual time frame from messages
						const firstMessageTime = getSendTime(results[0] as R);
						const lastMessageTime = getSendTime(results[results.length - 1] as R);

						// Determine the summary type and format the header
						let summaryHeader = "";
						if (summary.endsWith("h")) {
							const hours = parseInt(summary);
							summaryHeader = `Chat summary for the last ${hours} hour${hours === 1 ? '' : 's'}\nTime frame: ${firstMessageTime} to ${lastMessageTime}\nMessage count: ${results.length}`;
						} else {
							summaryHeader = `Chat summary of the last ${results.length} messages\nTime frame: ${firstMessageTime} to ${lastMessageTime}`;
						}

						await sendRichMessageDraft(
							env,
							groupId,
							bot.update.message!.message_id,
							`Summarizing ${results.length} message${results.length === 1 ? '' : 's'}. Standby...`,
						);

						const sent = await generateAndSendSummary(env, groupId, results, summaryHeader, bot.update.message!.message_id);
						if (!sent.ok) {
							await bot.reply(`⚠️ Unable to generate enhanced markdown.`);
						}
					}
					catch (e) {
						console.error(e);
						await bot.reply(`Summary failed: ${(e as Error).message}`);
					}
				}

				return new Response('ok');
			})
			.on("model", async (ctx) => {
				const chat = ctx.update.message?.chat;
				if (!chat || !chat.type.includes('group')) {
					await ctx.reply('Please use /model in a group chat.');
					return new Response('ok');
				}
				const groupId = chat.id;
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
					await ctx.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					const groupName = chat.title || 'Unknown';
					await notifyOwnerAboutGroup(ctx.api, env, groupId, groupName);
					return new Response('ok');
				}

				const messageText = (ctx.update.message?.text || "").trim();
				const arg = messageText.split(/\s+/)[1]?.trim();
				const current = await getGroupModelSelection(env, groupId);

				if (!arg || arg.toLowerCase() === "list") {
					await ctx.reply(
						`Current model: ${current.modelKey}\nAvailable models:\n${formatModelOptions().join("\n")}\n\nUse /model <model-key> to switch.\nCustom format: /model openai:<model> or /model google:<model> or /model anthropic:<model> or /model workers-ai:<model>`
					);
					return new Response('ok');
				}

				const requested = getModelByKey(arg);
				if (!requested) {
					await ctx.reply(
						`Unknown model: ${arg}\nAvailable models:\n${formatModelOptions().join("\n")}\n\nCustom format: /model openai:<model> or /model google:<model> or /model anthropic:<model> or /model workers-ai:<model>`
					);
					return new Response('ok');
				}
				if (requested.modelConfig.provider !== "workers-ai" && !env.WORKER_AI_TOKEN) {
					await ctx.reply("WORKER_AI_TOKEN is not configured. Add it before selecting gateway-routed models.");
					return new Response('ok');
				}
				if (requested.modelConfig.provider === "anthropic" && !env.ANTHROPIC_API_KEY) {
					await ctx.reply("ANTHROPIC_API_KEY is not configured. Add it before selecting Claude models.");
					return new Response('ok');
				}
				if (requested.modelConfig.provider === "google" && !env.GEMINI_API_KEY) {
					await ctx.reply("GEMINI_API_KEY is not configured. Add it before selecting Gemini models.");
					return new Response('ok');
				}

				await setGroupModelSelection(env, groupId, requested.modelKey, ctx.update.message?.from?.id);
				await ctx.reply(`Model updated to ${requested.modelKey} (${requested.modelConfig.label}).`);
				return new Response('ok');
			})
			.on("imagemodel", async (ctx) => {
				const chat = ctx.update.message?.chat;
				if (!chat || !chat.type.includes('group')) {
					await ctx.reply('Please use /imagemodel in a group chat.');
					return new Response('ok');
				}
				const groupId = chat.id;
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
					await ctx.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					const groupName = chat.title || 'Unknown';
					await notifyOwnerAboutGroup(ctx.api, env, groupId, groupName);
					return new Response('ok');
				}

				const messageText = (ctx.update.message?.text || "").trim();
				const arg = messageText.split(/\s+/)[1]?.trim();
				const current = await getGroupImageModelSelection(env, groupId);

				if (!arg || arg.toLowerCase() === "list") {
					await ctx.reply(
						`Current image model: ${current.modelKey}\nAvailable image models:\n${formatImageModelOptions().join("\n")}\n\nUse /imagemodel <model-key> to switch, or /imagemodel off to disable.\nCustom format: /imagemodel google:<model> or /imagemodel workers-ai:<model>`
					);
					return new Response('ok');
				}

				const requested = getImageModelByKey(arg);
				if (!requested) {
					await ctx.reply(
						`Unknown image model: ${arg}\nAvailable image models:\n${formatImageModelOptions().join("\n")}\n\nCustom format: /imagemodel google:<model> or /imagemodel workers-ai:<model>`
					);
					return new Response('ok');
				}
				if (requested.modelConfig?.provider === "google" && !env.GEMINI_API_KEY) {
					await ctx.reply("GEMINI_API_KEY is not configured. Add it before selecting Gemini image models.");
					return new Response('ok');
				}

				await setGroupImageModelSelection(env, groupId, requested.modelKey, ctx.update.message?.from?.id);
				await ctx.reply(
					requested.modelKey === IMAGE_MODEL_OFF
						? "Image generation disabled for /summary in this group."
						: `Image model updated to ${requested.modelKey} (${requested.modelConfig!.label}).`
				);
				return new Response('ok');
			})
			.on("schedule", async (ctx) => {
				// Configure the daily scheduled summary. Two entry points:
				// - In a whitelisted group chat: /schedule [HH:00|off] applies to that group.
				// - In the owner's private chat (admin interface): /schedule list shows every
				//   whitelisted group with its ID, and /schedule <groupId> <HH:00|off> sets it.
				const chat = ctx.update.message?.chat;
				const userId = ctx.update.message?.from?.id;
				const messageText = (ctx.update.message?.text || "").trim();
				const args = messageText.split(/\s+/).slice(1);

				const groupUsage = `Usage:\n/schedule <HH:00> — send a daily summary of the last 24h at that hour (US/Eastern, top of the hour)\n/schedule off — disable the daily summary`;

				if (chat?.type === 'private') {
					const ownerUserId = parseInt(env.OWNER_ID);
					if (userId !== ownerUserId) {
						await ctx.reply('You are not authorized to use this command.');
						return new Response('ok');
					}

					const adminUsage = `Usage:\n/schedule list — show all whitelisted groups and their schedules\n/schedule <groupId> <HH:00> — set that group's daily summary hour (US/Eastern)\n/schedule <groupId> off — disable that group's daily summary`;

					if (!args[0] || args[0].toLowerCase() === 'list') {
						await ensureScheduleSettingsTable(env);
						const { results: groups } = await env.DB.prepare(`
							SELECT w.groupId,
								COALESCE(
									(SELECT m.groupName FROM Messages m WHERE CAST(m.groupId AS INTEGER) = CAST(w.groupId AS INTEGER) ORDER BY m.timeStamp DESC LIMIT 1),
									w.groupName
								) AS groupName,
								s.hourEt
							FROM WhitelistedGroups w
							LEFT JOIN GroupScheduleSettings s ON CAST(s.groupId AS INTEGER) = CAST(w.groupId AS INTEGER)
							ORDER BY w.groupId
						`).all<{ groupId: string, groupName: string, hourEt: number | null }>();
						if (!groups || groups.length === 0) {
							await ctx.reply('No whitelisted groups yet.');
							return new Response('ok');
						}
						const lines = groups.map((g) =>
							`${g.groupName} (${g.groupId}) — ${g.hourEt == null ? 'no daily summary' : `daily summary at ${formatScheduleHour(g.hourEt)}`}`
						);
						await ctx.reply(`Whitelisted groups:\n${lines.join('\n')}\n\n${adminUsage}`);
						return new Response('ok');
					}

					const targetGroupId = parseInt(args[0]);
					if (isNaN(targetGroupId)) {
						await ctx.reply(`Invalid group ID: ${args[0]}\n\n${adminUsage}`);
						return new Response('ok');
					}
					const whitelisted = await env.DB.prepare(`
						SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ? LIMIT 1
					`).bind(targetGroupId).first();
					if (!whitelisted) {
						await ctx.reply(`Group ${targetGroupId} is not whitelisted. Use /schedule list to see available groups.`);
						return new Response('ok');
					}

					if (!args[1]) {
						const currentHour = await getGroupScheduleHour(env, targetGroupId);
						await ctx.reply(
							`Group ${targetGroupId}: ${currentHour == null ? 'no daily summary scheduled' : `daily summary at ${formatScheduleHour(currentHour)}`}\n\n${adminUsage}`
						);
						return new Response('ok');
					}
					if (args[1].toLowerCase() === 'off') {
						await deleteGroupSchedule(env, targetGroupId);
						await ctx.reply(`Daily summary disabled for group ${targetGroupId}.`);
						return new Response('ok');
					}
					const hour = parseScheduleHour(args[1]);
					if (hour == null) {
						await ctx.reply(`Invalid time: ${args[1]}. Use a top-of-the-hour time like 21:00 or 9.\n\n${adminUsage}`);
						return new Response('ok');
					}
					await setGroupScheduleHour(env, targetGroupId, hour, userId);
					await ctx.reply(`Daily summary for group ${targetGroupId} scheduled at ${formatScheduleHour(hour)} (covers the previous 24 hours).`);
					return new Response('ok');
				}

				if (!chat || !chat.type.includes('group')) {
					await ctx.reply('Please use /schedule in a group chat, or in a direct chat with the bot (admin).');
					return new Response('ok');
				}

				const groupId = chat.id;
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();
				if (!whitelistResults || whitelistResults.length === 0) {
					await ctx.reply(`This group (ID: ${groupId}) is not whitelisted. Please contact the bot owner.`);
					const groupName = chat.title || 'Unknown';
					await notifyOwnerAboutGroup(ctx.api, env, groupId, groupName);
					return new Response('ok');
				}

				if (!args[0]) {
					const currentHour = await getGroupScheduleHour(env, groupId);
					await ctx.reply(
						`${currentHour == null ? 'No daily summary scheduled for this group.' : `Daily summary scheduled at ${formatScheduleHour(currentHour)}.`}\n\n${groupUsage}`
					);
					return new Response('ok');
				}
				if (args[0].toLowerCase() === 'off') {
					await deleteGroupSchedule(env, groupId);
					await ctx.reply('Daily summary disabled for this group.');
					return new Response('ok');
				}
				const hour = parseScheduleHour(args[0]);
				if (hour == null) {
					await ctx.reply(`Invalid time: ${args[0]}. Use a top-of-the-hour time like 21:00 or 9.\n\n${groupUsage}`);
					return new Response('ok');
				}
				await setGroupScheduleHour(env, groupId, hour, userId);
				await ctx.reply(`Daily summary scheduled at ${formatScheduleHour(hour)}. It will cover the previous 24 hours and is skipped if the group had fewer than ${MIN_SCHEDULED_SUMMARY_MESSAGES} messages.`);
				return new Response('ok');
			})
			.on("persona", async (ctx) => {
				// Admin-only command, usable in a direct (private) chat with the bot.
				// Sets a universal personality prompt and/or model temperature for
				// every group's /summary and /ask.
				const ownerUserId = parseInt(env.OWNER_ID);
				const userId = ctx.update.message?.from?.id;
				const chat = ctx.update.message?.chat;

				if (userId !== ownerUserId) {
					await ctx.reply('You are not authorized to use this command.');
					return new Response('ok');
				}
				if (chat?.type !== 'private') {
					await ctx.reply('Please use /persona in a direct chat with the bot.');
					return new Response('ok');
				}

				const text = (ctx.update.message?.text || "").trim();
				const firstSpace = text.indexOf(" ");
				const rest = firstSpace === -1 ? "" : text.slice(firstSpace + 1).trim();

				const showCurrent = async () => {
					const persona = await getBotPersona(env);
					const temp = await getBotTemperature(env);
					await ctx.reply(
						`Current personality prompt:
${persona ?? `(default) ${DEFAULT_PERSONA_SUMMARIZE}`}

` +
						`Temperature: ${temp}

` +
						`Usage:
` +
						`/persona <text> — set the personality prompt (applies to /summary and /ask in all groups)
` +
						`/persona temp <0-2> — set the model temperature
` +
						`/persona reset — restore the default persona and temperature`
					);
				};

				if (!rest) {
					await showCurrent();
					return new Response('ok');
				}

				const subSpace = rest.indexOf(" ");
				const sub = (subSpace === -1 ? rest : rest.slice(0, subSpace)).toLowerCase();
				const subArg = subSpace === -1 ? "" : rest.slice(subSpace + 1).trim();

				if (sub === "reset") {
					await deleteBotSetting(env, "persona");
					await deleteBotSetting(env, "temperature");
					await ctx.reply(`Persona and temperature reset to defaults (temperature ${DEFAULT_TEMPERATURE}).`);
					return new Response('ok');
				}

				if (sub === "temp" || sub === "temperature") {
					const value = parseFloat(subArg);
					if (!Number.isFinite(value) || value < 0 || value > 2) {
						await ctx.reply('Please provide a temperature between 0 and 2, e.g. /persona temp 0.7');
						return new Response('ok');
					}
					await setBotSetting(env, "temperature", String(value), userId);
					await ctx.reply(`Temperature updated to ${value}.`);
					return new Response('ok');
				}

				// Otherwise treat the whole argument as the new persona text.
				const persona = sub === "set" ? subArg : rest;
				if (!persona) {
					await ctx.reply('Please provide the personality text, e.g. /persona You are a witty pirate assistant.');
					return new Response('ok');
				}
				await setBotSetting(env, "persona", persona, userId);
				await ctx.reply(`Personality prompt updated. It now applies to /summary and /ask in all groups.

${persona}`);
				return new Response('ok');
			})
			.on('my_chat_member', async (ctx) => {
				const myChatMemberUpdate = (ctx.update as any).my_chat_member;
				console.debug('my_chat_member event triggered:', myChatMemberUpdate);
				// Triggered when bot is added/removed from a group
				const my_chat_member = myChatMemberUpdate;
				const groupId = my_chat_member.chat.id;
				const groupName = my_chat_member.chat.title || 'Unknown';
				const newStatus = my_chat_member.new_chat_member.status;

				console.debug(`Bot status change: ${my_chat_member.old_chat_member.status} -> ${newStatus} in group ${groupId} (${groupName})`);

				// If bot was just added to the group
				if ((my_chat_member.old_chat_member.status === 'left' || my_chat_member.old_chat_member.status === 'kicked') && 
				    (newStatus === 'member' || newStatus === 'administrator')) {
					// Send message to owner asking for whitelist
					const ownerUserId = parseInt(env.OWNER_ID);
					console.debug(`Attempting to notify owner ${ownerUserId} about group ${groupId}`);
					try {
						await (ctx.api as any).sendMessage(ownerUserId,
							`Bot added to new group: ${groupName} (ID: ${groupId})\n\nUse /whitelist ${groupId} to approve this group for processing.`
						);
						console.debug('Owner notification sent successfully');
					} catch (e) {
						console.error('Failed to notify owner:', e);
					}
				}
				return new Response('ok');
			})
			.on('whitelist', async (ctx) => {
				const ownerUserId = parseInt(env.OWNER_ID);
				const userId = ctx.update.message!.from!.id;

				// Only owner can use this command
				if (userId !== ownerUserId) {
					await ctx.reply('You are not authorized to use this command.');
					return new Response('ok');
				}

				const messageText = ctx.update.message!.text || '';
				const groupIdStr = messageText.split(' ')[1];

let chosenGroupId = groupIdStr;
			if (!chosenGroupId) {
				// If command is run inside a group (owner can do this), use that group id
				const chat = ctx.update.message?.chat;
				if (chat && chat.id && chat.type && chat.type.includes('group')) {
					chosenGroupId = String(chat.id);
				}
			}

			if (!chosenGroupId) {
				await ctx.reply('Usage: /whitelist <groupId>');
				return new Response('ok');
			}

			const groupId = chosenGroupId;
			try {
				// Add to whitelist (ensure numeric ID)
				const numericGroupId = parseInt(groupId);
				if (isNaN(numericGroupId)) {
					await ctx.reply('Invalid group ID. Must be a number.');
					return new Response('ok');
				}

				await env.DB.prepare(`
					INSERT OR REPLACE INTO WhitelistedGroups(groupId, groupName, whitelistedAt)
					VALUES (CAST(? AS INTEGER), ?, ?)
				`).bind(numericGroupId, `Group ${numericGroupId}`, Date.now()).run();

				await ctx.reply(`Group ${numericGroupId} has been whitelisted!`);
				} catch (e) {
					console.error('Whitelist error:', e);
					await ctx.reply('Error whitelisting group. Make sure the group ID is correct.');
				}
				return new Response('ok');
			})
			.on(':message', async (bot) => {
				if (!bot.update.message!.chat.type.includes('group')) {
					await bot.reply('I am a bot, please add me to a group to use me.');
					return new Response('ok');
				}

			const groupId = bot.update.message!.chat.id; // numeric ID

			// Check if group is whitelisted
			const { results: whitelistResults } = await env.DB.prepare(`
				SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
			`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
					// Group not whitelisted, ignore message but notify owner
					console.debug(`Message from non-whitelisted group: ${groupId}`);
					const groupName = bot.update.message!.chat.title || 'Unknown';
					const messageText = bot.update.message!.text || '';
					// Only notify for non-command messages to avoid spam
					if (messageText && !messageText.startsWith('/')) {
						await notifyOwnerAboutGroup(bot.api, env, groupId, groupName);
					}
					return new Response('ok');
				}

				switch (bot.update_type) {
					case 'message': {
						const msg = bot.update.message!;
						const groupId = msg.chat.id;
						let content = msg.text || "";
						const fwd = msg.forward_from?.last_name;
						const replyTo = msg.reply_to_message?.message_id;
						if (fwd) {
							content = `Forwarded from ${fwd}: ${content}`;
						}
						if (replyTo) {
							content = `Reply to ${getMessageLink({ groupId: groupId.toString(), messageId: replyTo })}: ${content}`;
						}
						if (content.startsWith("http") && !content.includes(" ")) {
							content = await extractAllOGInfo(content);
						}
						const messageId = msg.message_id;
						const groupName = msg.chat.title || "anonymous";
						const timeStamp = Date.now();
						const userName = getUserName(msg);
						try {
							await env.DB.prepare(`
								INSERT INTO Messages(id, groupId, timeStamp, userName, content, messageId, groupName) VALUES (?, ?, ?, ?, ?, ?, ?)`)
								.bind(
									getMessageLink({ groupId: groupId.toString(), messageId }),
									groupId,
									timeStamp,
									userName, // not interested in user id
									content,
									messageId,
									groupName
								)
								.run();
						}
						catch (e) {
							console.error(e);
						}
						return new Response('ok');

					}
					case "photo": {
						const msg = bot.update.message!;
						const groupId = msg.chat.id;
						const messageId = msg.message_id;
						const groupName = msg.chat.title || "anonymous";
						const timeStamp = Date.now();
						const userName = getUserName(msg);
						const photo = msg.photo![msg.photo!.length - 1];
						const file = await bot.getFile(photo.file_id).then((response) => response.arrayBuffer());
						if (!(isJPEGBase64(Buffer.from(file).toString("base64")).isValid)) {
							console.error("not a jpeg");
							return new Response('ok');
						}
						try {
							await env.DB.prepare(`
							INSERT OR REPLACE INTO Messages(id, groupId, timeStamp, userName, content, messageId, groupName) VALUES (?, ?, ?, ?, ?, ?, ?)`)
								.bind(
									getMessageLink({ groupId: groupId.toString(), messageId }),
									groupId,
									timeStamp,
									userName, // not interested in user id
									"data:image/jpeg;base64," + Buffer.from(file).toString("base64"),
									messageId,
									groupName
								)
								.run();
						}
						catch (e) {
							console.error(e);
						}
						return new Response('ok');
					}
				}
				return new Response('ok');
			})
			.on(":edited_message", async (ctx) => {
				const msg = ctx.update.edited_message!;
				const groupId = msg.chat.id;
				
				// Check if group is whitelisted
				const { results: whitelistResults } = await env.DB.prepare(`
					SELECT groupId FROM WhitelistedGroups WHERE CAST(groupId AS INTEGER) = ?
				`).bind(groupId).all();

				if (!whitelistResults || whitelistResults.length === 0) {
					// Group not whitelisted, ignore edited message
					console.debug(`Edited message from non-whitelisted group: ${groupId}`);
					return new Response('ok');
				}
				
				const content = msg.text || "";
				const messageId = msg.message_id;
				const groupName = msg.chat.title || "anonymous";
				const timeStamp = Date.now();
				const userName = getUserName(msg);
				try {
					await env.DB.prepare(`
					INSERT OR REPLACE INTO Messages(id, groupId, timeStamp, userName, content, messageId, groupName) VALUES (?, ?, ?, ?, ?, ?, ?)`)
						.bind(
							getMessageLink({ groupId: groupId.toString(), messageId }),
							groupId,
							timeStamp,
							userName, // not interested in user id
							content,
							messageId,
							groupName
						)
						.run();
				}
				catch (e) {
					console.error(e);
				}
				return new Response('ok');
			})
			.handle(request.clone());
		return new Response('ok');
	},
};
