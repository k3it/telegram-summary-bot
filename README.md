# telegram group summary bot

![image_fx_](https://github.com/user-attachments/assets/d9924ced-5310-4edc-9073-bdcc1df1dd6d)

> Image by imagen-3 on ImageFX, prompt "an icon for telegram group message summarize and search message bot sketchy"

Feel scared when seeing 2000+ unread in groups? Now you can read them by LLM!

Use d1, cf worker, ai gateway, gemini-3.5-flash.

## NOTICE

Due to current usage exceeded maximum DB size 500M `D1_ERROR: Exceeded maximum DB size`, please deploy your own bot. And you can keep a much longer log. I have disabled this bot to be add to new group.

## Setup

bot: <https://github.com/codebam/cf-workers-telegram-bot>

d1: <https://developers.cloudflare.com/d1/get-started/>

check wiki

All model calls (OpenAI/Google/Anthropic) are routed through Cloudflare AI Gateway; set `WORKER_AI_TOKEN` (a Cloudflare API token), `CF_ACCOUNT_ID`, and `AI_GATEWAY_ID`. Set `ANTHROPIC_API_KEY` if you want to use Claude models. `@cf/*` Workers AI models route via the `AI` binding directly, no key needed.

`/summary` illustrations are generated then uploaded to the `SUMMARY_IMAGES` R2 bucket (keyed `<groupId>/<random14charstring>.<ext>`) and embedded via its public r2.dev URL (`R2_PUBLIC_URL`) — Telegram's `sendRichMessage` doesn't support attaching raw file bytes, only a real HTTPS image URL in markdown. The r2.dev dev URL is rate-limited; switch to a custom domain on the bucket if a group's summary traffic outgrows it.

## Usage

/summary 10

summarize newest 10 messages

/summary 10h


summarize newest 10h messages

/query word

can search cjk

/ask question

answer question based on group chat

/status

check bot is alive

/model

show current model and available options for this group

/model claude-3.7-sonnet

switch model for this group (persisted in D1)

/model google:gemini-2.5-flash

set a custom provider model id (`openai:...`, `google:...`, `anthropic:...`, `workers-ai:...`)

/imagemodel

show current image model and available options for this group (used by `/summary` to illustrate the summary)

/imagemodel nano-banana-2

switch image model for this group (persisted in D1)

/imagemodel off

disable image generation for `/summary` in this group

## Cost

**0**

d1: telegram bot can't read info in the history, only newly sent info, so use database to keep them. [pricing](https://developers.cloudflare.com/d1/platform/pricing/#billing-metrics)

cf worker: so no more offline.(I hope so) [pricing](https://developers.cloudflare.com/workers/platform/pricing/#workers)

gemini-3.5-flash: free for limited usage and huge context

---

The Newest 4000 text messages will be kept, images will be kept for 2 days. May change in the future.

---

## TODO

~~add method to import group history information so history info can be searched.~~

maybe auto-delte query result

---

TODO:

maybe word cloud?

## Limitation

This bot will store chat history into my d1 database, so deploy your own bot is recommended. ~~Unless I am in your group~~.

Only messages sent by **user** and **after** the bot join the group can be summarized/query.

> Why doesn't my bot see messages from other bots? Bots talking to each other could potentially get stuck in unwelcome loops. To avoid this, we decided that bots will not be able to see messages from other bots regardless of mode.
>
> https://core.telegram.org/bots/faq#why-doesn-39t-my-bot-see-messages-from-other-bots

> You can't get older message than bot join.
>
> https://github.com/yagop/node-telegram-bot-api/issues/577

If you want to bypass these, check [luoxu](https://github.com/lilydjwg/luoxu), which uses userbot.
