import TelegramBot, { TelegramApi } from '@codebam/cf-workers-telegram-bot';
import OpenAI from "openai";

import telegramifyMarkdown from "telegramify-markdown"
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

type Provider = "openai" | "anthropic" | "google";
type ModelConfig = {
	provider: Provider;
	model: string;
	label: string;
	noTemperature?: boolean;
};

const DEFAULT_MODEL_KEY = "gpt-5.4-nano";
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
	"gemini-2.5-pro": { provider: "google", model: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
	"gemini-2.5-flash": { provider: "google", model: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
	"gemini-2.5-flash-lite": { provider: "google", model: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite" },
	"claude-3.7-sonnet": { provider: "anthropic", model: "claude-3-7-sonnet-latest", label: "Claude 3.7 Sonnet" },
	"claude-3.5-sonnet": { provider: "anthropic", model: "claude-3-5-sonnet-latest", label: "Claude 3.5 Sonnet" },
	"claude-3.5-haiku": { provider: "anthropic", model: "claude-3-5-haiku-latest", label: "Claude 3.5 Haiku" },
};

const temperature = 0.4;

let modelSettingsTableReady = false;

function normalizeModelKey(input: string) {
	return input.trim().toLowerCase();
}

function getModelByKey(modelKey: string) {
	const normalized = normalizeModelKey(modelKey);
	const modelConfig = MODEL_REGISTRY[normalized];
	if (!modelConfig) {
		const customMatch = normalized.match(/^(openai|google|anthropic):(.+)$/);
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
	];
}

function getGenModel(env: Env) {
	const openai = new OpenAI({
		apiKey: env.OPENAI_API_KEY,
		timeout: 999999999999,
	});
	return openai;
}

function getGoogleGenModel(env: Env) {
	const google = new OpenAI({
		apiKey: env.GEMINI_API_KEY,
		baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
		timeout: 999999999999,
	});
	return google;
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

type DispatchContent = ReturnType<typeof dispatchContent>;
type ChatMessage = {
	role: "system" | "user" | "assistant";
	content: string | DispatchContent[];
};

type AnthropicContentBlock =
	| { type: "text"; text: string }
	| {
		type: "image";
		source: {
			type: "base64";
			media_type: string;
			data: string;
		};
	};

function toAnthropicContent(content: string | DispatchContent[]): AnthropicContentBlock[] {
	if (typeof content === "string") {
		return [{ type: "text", text: content }];
	}
	const blocks: AnthropicContentBlock[] = [];
	for (const block of content) {
		if (block.type === "text") {
			blocks.push({ type: "text", text: block.text });
			continue;
		}
		const imageUrl = block.image_url.url;
		const dataUrlMatch = imageUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
		if (!dataUrlMatch) {
			continue;
		}
		blocks.push({
			type: "image",
			source: {
				type: "base64",
				media_type: dataUrlMatch[1],
				data: dataUrlMatch[2],
			},
		});
	}
	return blocks;
}

async function createModelResponse(
	env: Env,
	selectedModel: { modelKey: string, modelConfig: ModelConfig },
	messages: ChatMessage[],
	maxTokens = 4096,
) {
	if (selectedModel.modelConfig.provider === "openai" || selectedModel.modelConfig.provider === "google") {
		if (selectedModel.modelConfig.provider === "google" && !env.GEMINI_API_KEY) {
			throw new Error("GEMINI_API_KEY is not configured.");
		}
		const client = selectedModel.modelConfig.provider === "google" ? getGoogleGenModel(env) : getGenModel(env);
		const response = await client.chat.completions.create({
			model: selectedModel.modelConfig.model,
			messages: messages as any,
			max_completion_tokens: maxTokens,
			...(selectedModel.modelConfig.noTemperature ? {} : { temperature }),
		});
		return response.choices[0].message.content || "";
	}

	if (!env.ANTHROPIC_API_KEY) {
		throw new Error("ANTHROPIC_API_KEY is not configured.");
	}

	const systemContent = messages
		.filter((m) => m.role === "system")
		.map((m) => (typeof m.content === "string" ? m.content : ""))
		.filter(Boolean)
		.join("\n\n");
	const conversation = messages
		.filter((m) => m.role !== "system")
		.map((m) => ({
			role: m.role,
			content: toAnthropicContent(m.content),
		}));

	const response = await fetch("https://api.anthropic.com/v1/messages", {
		method: "POST",
		headers: {
			"x-api-key": env.ANTHROPIC_API_KEY,
			"anthropic-version": "2023-06-01",
			"content-type": "application/json",
		},
		body: JSON.stringify({
			model: selectedModel.modelConfig.model,
			system: systemContent,
			messages: conversation,
			max_tokens: maxTokens,
			temperature,
		}),
	});
	if (!response.ok) {
		throw new Error(`Anthropic request failed: ${response.status} ${await response.text()}`);
	}
	const data = await response.json<any>();
	const text = (data?.content || [])
		.filter((c: any) => c?.type === "text")
		.map((c: any) => c?.text || "")
		.join("\n")
		.trim();
	return text;
}

function foldText(text: string) {
	return text.split("\n").map((line) => '>' + line).join("\n");
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

// System prompts for different scenarios
const SYSTEM_PROMPTS = {
  summarizeChat: `You are a professional group chat summarization assistant. Your task is to summarize conversations in a natural, group-chat-friendly tone, in English only.

The conversation will be provided in the following format:
====================
Username:
Message content
Associated link
====================

Follow these guidelines:
1. If multiple topics are discussed, summarize them as separate sections in the summary, clearly indicating topic shifts
2. If images are mentioned, include relevant descriptions in the summary
3. Use markdown format to reference original messages with links
4. Link format should be: [Keyword1](URL), etc.   prefer to use keyword as link text if possible, but if not, just use Ref + number.
5. Keep the summary concise while capturing key content and sentiment
6. Start the summary with the time frame and message count information provided
7. Output must be entirely in English, but ok to include non-English content from the chat in the summary as long as the summary itself is in English
8. For each section of the summary add a very brief AI opinion on the discussion, but clearly indicate that it's an opinion from the AI
9. Use proper markdown formatting to enhance readability, but avoid nested bullet lists and avoid blockquotes.
10. Keep the total response within 256 words, and try to be as concise as possible while following the above guidelines.
11. Use this structure for readability:
    - First line: one compact time/message-count line.
    - Then sections titled exactly like: "Topic 1: ...", "Topic 2: ..."
    - For each topic, use normal paragraph flow with short paragraphs and inline links.
    - Put only the single "AI opinion:" line in a fenced code block (\`\`\` ... \`\`\`).
    - Do not separate links into their own bullet list; flow them naturally with the relevant sentence.
12. Keep lines natural sentence flow; do not break every sentence into a new line.`,

  answerQuestion: `You are an intelligent group chat assistant. Your task is to answer user questions based on the provided chat history, in English only.

The chat history will be provided in the following format:
====================
Username:
Message content
Associated link
====================

1. If multiple topics are discussed, summarize them as separate sections in the summary, clearly indicating topic shifts
2. If images are mentioned, include relevant descriptions in the summary
3. Use markdown format to reference original messages with links
4. Link format should be: [Keyword1](URL), etc.   prefer to use keyword as link text if possible, but if not, just use Ref + number.
5. Keep the summary concise while capturing key content and sentiment
6. Start the summary with the time frame and message count information provided
7. Output must be entirely in English, but ok to include non-English content from the chat in the summary as long as the summary itself is in English
9. Use proper markdown formatting to enhance readability, but avoid nested bullet lists and avoid blockquotes.
10. Keep the total response within 256 words, and try to be as concise as possible while following the above guidelines.
11. Prefer compact paragraphs over list-heavy formatting unless the user explicitly asks for bullets.`
};

function getCommandVar(str: string, delim: string) {
	return str.slice(str.indexOf(delim) + delim.length);
}

function messageTemplate(s: string, modelName: string) {
	return `Summary by ${escapeMarkdownV2(modelName)}\n` + s;
}

function splitTelegramMessage(text: string, maxLen = 3900) {
	const chunks: string[] = [];
	let remaining = text;
	while (remaining.length > maxLen) {
		let splitIndex = remaining.lastIndexOf('\n', maxLen);
		if (splitIndex <= 0) {
			splitIndex = maxLen;
		}
		chunks.push(remaining.slice(0, splitIndex));
		remaining = remaining.slice(splitIndex);
	}
	if (remaining.length > 0) {
		chunks.push(remaining);
	}
	return chunks;
}

async function sendSummaryText(bot: any, text: string, fallbackRawText?: string) {
	const chunks = splitTelegramMessage(text, 3900);
	for (const chunk of chunks) {
		const res = await bot.reply(chunk, 'MarkdownV2');
		if (!res?.ok) {
			const body = await res.json().catch(() => null);
			const description = body?.description || '';
			if (description.includes("can't parse entities") || description.includes('message is too long')) {
				// Fallback to fully escaped MarkdownV2 text to keep output readable and stable.
				const safeText = escapeMarkdownV2(fallbackRawText ?? text);
				const safeChunks = splitTelegramMessage(safeText, 3900);
				for (const safeChunk of safeChunks) {
					const fallbackRes = await bot.reply(safeChunk, 'MarkdownV2');
					if (!fallbackRes?.ok) {
						console.error('Fallback safe MarkdownV2 reply failed', await fallbackRes?.text());
					}
				}
				return;
			}
			console.error('Failed to send reply', res?.statusText, await res?.text());
			return;
		}
	}
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

function formatSummaryAsTopicCards(text: string) {
	const sanitized = text
		// Remove any existing markdown fences the model may emit (escaped or unescaped).
		.replace(/^\s*```[^\n]*\s*$/gm, "")
		.replace(/^\s*\\`\\`\\`[^\n]*\s*$/gm, "")
		.replace(/^\s*(?:\\`){3}[^\n]*\s*$/gm, "");
	const lines = sanitized.replace(/\r\n/g, "\n").split("\n");
	const topicHeaderRegex = /^\*?\s*Topic\s+\d+\s*:/i;
	const topicIndexes: number[] = [];

	for (let i = 0; i < lines.length; i++) {
		if (topicHeaderRegex.test(lines[i].trim())) {
			topicIndexes.push(i);
		}
	}

	if (topicIndexes.length === 0) {
		return text;
	}

	const intro = lines.slice(0, topicIndexes[0]).join("\n").trim();
	const sections: string[] = [];

	for (let i = 0; i < topicIndexes.length; i++) {
		const start = topicIndexes[i];
		const end = i + 1 < topicIndexes.length ? topicIndexes[i + 1] : lines.length;
		const chunk = lines.slice(start, end);
		const header = (chunk[0] || "").trim();
		const bodyLines = chunk.slice(1);

		let body = bodyLines.join("\n").trim();
		if (!body) {
			body = "No additional details.";
		}
		body = body.replace(/```/g, "'''");
		const headerNoStars = header.replace(/^\*+/, "").replace(/\*+$/, "").trim();
		const renderHeader = `*${headerNoStars}*`;

		// Keep normal flow, but render AI opinion as a code block for visual emphasis.
		const aiOpinionPattern = /(^|\n)(\*?\s*AI opinion:\*?\s*[^\n]*)/gi;
		const bodyWithCodeOpinion = body.replace(aiOpinionPattern, (_m, prefix, opinionLine) => {
			const unescaped = opinionLine
				.replace(/\\([_\*\[\]\(\)~`>#+\-=|{}.!])/g, "$1")
				.replace(/\\\\/g, "\\");
			return `${prefix}\`\`\`\n${unescaped}\n\`\`\``;
		});

		sections.push(`${renderHeader}\n\n${bodyWithCodeOpinion}`.trim());
	}

	return `${intro}\n\n${sections.join("\n\n")}`.trim();
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
					"parse_mode": "MarkdownV2",
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
					answerText = await createModelResponse(
						env,
						selectedModel,
						[
							{
								role: "system",
								content: SYSTEM_PROMPTS.answerQuestion,
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
					);
				} catch (e) {
					console.error(e);
					await ctx.reply(`Model call failed: ${(e as Error).message}`);
					return new Response('ok');
				}
				let response_text: string;
				response_text = processMarkdownLinks(telegramifyMarkdown(answerText || "", 'keep'));

				res = await ctx.api.sendMessage(ctx.bot.api.toString(), {
					"chat_id": userId,
					"parse_mode": "MarkdownV2",
					"text": response_text,
					reply_to_message_id: -1,
				});
				if (!res.ok) {
					let reason = (await res.json() as any)?.promptFeedback?.blockReason;
					if (reason) {
						await ctx.reply(`Unable to answer, reason ${reason}`);
						return new Response('ok');
					}
					await ctx.reply(`Send failed`);
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
						const selectedModel = await getGroupModelSelection(env, groupId);
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

						const rawSummary = await createModelResponse(
							env,
							selectedModel,
							[
								{
									role: "system",
									content: SYSTEM_PROMPTS.summarizeChat,
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
						);

						const summaryText = messageTemplate(
							formatSummaryAsTopicCards(
								fixLink(
									processMarkdownLinks(telegramifyMarkdown(rawSummary, 'escape')))),
							selectedModel.modelKey);
						await sendSummaryText(bot, summaryText, `Summary by ${selectedModel.modelKey}\n${rawSummary}`);
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
						`Current model: ${current.modelKey}\nAvailable models:\n${formatModelOptions().join("\n")}\n\nUse /model <model-key> to switch.\nCustom format: /model openai:<model> or /model google:<model> or /model anthropic:<model>`
					);
					return new Response('ok');
				}

				const requested = getModelByKey(arg);
				if (!requested) {
					await ctx.reply(
						`Unknown model: ${arg}\nAvailable models:\n${formatModelOptions().join("\n")}\n\nCustom format: /model openai:<model> or /model google:<model> or /model anthropic:<model>`
					);
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
