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
} from "./../src/index"

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
	it("resolves the default gemini-3.5-flash registry entry", () => {
		const result = getModelByKey("gemini-3.5-flash");
		expect(result?.modelConfig.provider).toBe("google");
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
