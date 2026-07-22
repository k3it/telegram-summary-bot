// test/index.spec.ts
import { describe, it, expect } from 'vitest';
import {
	processMarkdownLinks,
	toSuperscript,
	getModelByKey,
	getImageModelByKey,
	GATEWAY_PROVIDER_SLUG,
	extractInlineImage,
	randomAlphanumeric,
	parseScheduleHour,
	extractRefsFooter,
	buildContinuityNote,
	mergeUsedRefs,
	resolveMaxOutputTokens,
} from "./../src/index"

describe("resolveMaxOutputTokens", () => {
	it("uses a valid numeric override", () => {
		expect(resolveMaxOutputTokens("8192", 32768)).toBe(8192);
	});

	it("falls back when the override is missing or unparseable", () => {
		expect(resolveMaxOutputTokens(undefined, 32768)).toBe(32768);
		expect(resolveMaxOutputTokens("", 32768)).toBe(32768);
		expect(resolveMaxOutputTokens("lots", 32768)).toBe(32768);
	});

	it("rejects zero and negative budgets rather than starving the reply", () => {
		expect(resolveMaxOutputTokens("0", 32768)).toBe(32768);
		expect(resolveMaxOutputTokens("-5", 32768)).toBe(32768);
	});
})

describe("test fix link", () => {
	it("should fix link", () => {
		const markdown = `
		This is a test text
		[link11111](link11111)     // identical, will be processed
		[link11111](link11112221)  // not identical, keep as is
		[another text](link11112221) // not same, keep as is
		[link22222](link22222)     // identical, will be processed
		[link11111](link11111)     // identical, will reuse number
		`;
		const result = processMarkdownLinks(markdown);
		expect(result).toBe(`
		This is a test text
		[reference¹](link11111)     // identical, will be processed
		[link11111](link11112221)  // not identical, keep as is
		[another text](link11112221) // not same, keep as is
		[reference²](link22222)     // identical, will be processed
		[reference¹](link11111)     // identical, will reuse number
		`);
	})
})
describe("upper number", () => {
	it("should upper number", () => {
		expect(toSuperscript(1234)).toBe("¹²³⁴");
	})
}
)

describe("gateway provider slug mapping", () => {
	it("maps registry provider names to AI Gateway compat slugs", () => {
		expect(GATEWAY_PROVIDER_SLUG.openai).toBe("openai");
		expect(GATEWAY_PROVIDER_SLUG.google).toBe("google-ai-studio");
		expect(GATEWAY_PROVIDER_SLUG.anthropic).toBe("anthropic");
	});
});

describe("getModelByKey", () => {
	it("resolves the default gemini-3.6-flash registry entry", () => {
		const result = getModelByKey("gemini-3.6-flash");
		expect(result?.modelConfig.provider).toBe("google");
		expect(result?.modelConfig.model).toBe("gemini-3.6-flash");
	});

	it("keeps gemini-3.5-flash selectable after the 3.6 default switch", () => {
		const result = getModelByKey("gemini-3.5-flash");
		expect(result?.modelConfig.model).toBe("gemini-3.5-flash");
	});

	it("parses custom provider:model syntax, including workers-ai", () => {
		const result = getModelByKey("workers-ai:@cf/meta/llama-4-scout-17b-16e-instruct");
		expect(result?.modelKey).toBe("workers-ai:@cf/meta/llama-4-scout-17b-16e-instruct");
		expect(result?.modelConfig.provider).toBe("workers-ai");
		expect(result?.modelConfig.model).toBe("@cf/meta/llama-4-scout-17b-16e-instruct");
	});

	it("returns null for unknown keys with no custom-syntax match", () => {
		expect(getModelByKey("not-a-real-model")).toBeNull();
	});
});

describe("getImageModelByKey", () => {
	it("resolves the off sentinel to a null modelConfig", () => {
		const result = getImageModelByKey("off");
		expect(result?.modelKey).toBe("off");
		expect(result?.modelConfig).toBeNull();
	});

	it("resolves the default nano-banana-2-lite registry entry", () => {
		const result = getImageModelByKey("nano-banana-2-lite");
		expect(result?.modelConfig?.provider).toBe("google");
		expect(result?.modelConfig?.model).toBe("gemini-3.1-flash-lite-image");
	});

	it("parses custom google:/workers-ai: syntax", () => {
		const result = getImageModelByKey("workers-ai:@cf/black-forest-labs/flux-1-schnell");
		expect(result?.modelConfig?.provider).toBe("workers-ai");
		expect(result?.modelConfig?.model).toBe("@cf/black-forest-labs/flux-1-schnell");
	});

	it("returns null for unknown keys", () => {
		expect(getImageModelByKey("not-a-real-image-model")).toBeNull();
	});
});

describe("extractInlineImage", () => {
	it("pulls base64 image data out of a canned generateContent response", () => {
		const canned = {
			candidates: [
				{
					content: {
						parts: [
							{ text: "here you go" },
							{ inlineData: { mimeType: "image/png", data: "aGVsbG8=" } },
						],
					},
				},
			],
		};
		expect(extractInlineImage(canned)).toEqual({ mimeType: "image/png", data: "aGVsbG8=" });
	});

	it("returns null when no inline image part is present", () => {
		const canned = { candidates: [{ content: { parts: [{ text: "no image here" }] } }] };
		expect(extractInlineImage(canned)).toBeNull();
	});

	it("returns null for malformed/empty responses", () => {
		expect(extractInlineImage({})).toBeNull();
		expect(extractInlineImage(null)).toBeNull();
	});
});

describe("parseScheduleHour", () => {
	it("accepts bare hours and HH:00 forms", () => {
		expect(parseScheduleHour("21")).toBe(21);
		expect(parseScheduleHour("21:00")).toBe(21);
		expect(parseScheduleHour("9:00")).toBe(9);
		expect(parseScheduleHour("09:00")).toBe(9);
		expect(parseScheduleHour("0")).toBe(0);
		expect(parseScheduleHour("23:00")).toBe(23);
	});

	it("rejects non-top-of-hour minutes (cron fires hourly)", () => {
		expect(parseScheduleHour("21:30")).toBeNull();
		expect(parseScheduleHour("9:15")).toBeNull();
	});

	it("rejects out-of-range and malformed input", () => {
		expect(parseScheduleHour("24")).toBeNull();
		expect(parseScheduleHour("25:00")).toBeNull();
		expect(parseScheduleHour("-1")).toBeNull();
		expect(parseScheduleHour("off")).toBeNull();
		expect(parseScheduleHour("21:0")).toBeNull();
		expect(parseScheduleHour("")).toBeNull();
	});
});

describe("extractRefsFooter", () => {
	it("strips the footer and parses semicolon-separated refs", () => {
		const raw = `**Topic 1: Deploys**\nStuff happened.\n\n<!--refs: The Matrix; boiling frog analogy-->`;
		const { summary, refs } = extractRefsFooter(raw);
		expect(summary).toBe("**Topic 1: Deploys**\nStuff happened.");
		expect(refs).toEqual(["The Matrix", "boiling frog analogy"]);
	});

	it("treats a 'none' footer as no refs", () => {
		const { summary, refs } = extractRefsFooter("Summary text.\n<!--refs: none-->");
		expect(summary).toBe("Summary text.");
		expect(refs).toEqual([]);
	});

	it("tolerates a missing footer", () => {
		const { summary, refs } = extractRefsFooter("Summary with no footer.");
		expect(summary).toBe("Summary with no footer.");
		expect(refs).toEqual([]);
	});

	it("ignores refs-like comments that are not the final line", () => {
		const raw = "Before <!--refs: Dune--> after.";
		const { summary, refs } = extractRefsFooter(raw);
		expect(summary).toBe(raw);
		expect(refs).toEqual([]);
	});

	it("drops empty items and trailing whitespace variants", () => {
		const { refs } = extractRefsFooter("Text\n<!-- refs: Dune; ; 1984 -->\n");
		expect(refs).toEqual(["Dune", "1984"]);
	});
});

describe("buildContinuityNote", () => {
	it("returns null when there is no history", () => {
		expect(buildContinuityNote([], [])).toBeNull();
	});

	it("includes previous summaries between markers", () => {
		const note = buildContinuityNote(["newest summary", "older summary"], []);
		expect(note).toContain("<previous-summaries>");
		expect(note).toContain("newest summary\n\n--- earlier summary ---\n\nolder summary");
		expect(note).not.toContain("already used these references");
	});

	it("includes the used-refs blocklist without summaries", () => {
		const note = buildContinuityNote([], ["The Matrix", "Dune"]);
		expect(note).toContain("The Matrix; Dune");
		expect(note).not.toContain("<previous-summaries>");
	});
});

describe("mergeUsedRefs", () => {
	it("appends new refs and dedupes case-insensitively", () => {
		expect(mergeUsedRefs(["The Matrix"], ["the matrix", "Dune"])).toEqual(["The Matrix", "Dune"]);
	});

	it("drops the oldest refs past the cap", () => {
		expect(mergeUsedRefs(["a", "b", "c"], ["d", "e"], 4)).toEqual(["b", "c", "d", "e"]);
	});
});

describe("randomAlphanumeric", () => {
	it("defaults to a 14-char alphanumeric string", () => {
		const key = randomAlphanumeric();
		expect(key).toHaveLength(14);
		expect(key).toMatch(/^[A-Za-z0-9]{14}$/);
	});

	it("honors a custom length", () => {
		expect(randomAlphanumeric(8)).toHaveLength(8);
	});

	it("is not deterministic across calls (unguessable R2 keys)", () => {
		const keys = new Set(Array.from({ length: 20 }, () => randomAlphanumeric()));
		expect(keys.size).toBe(20);
	});
});
