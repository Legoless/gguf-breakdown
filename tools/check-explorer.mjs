#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { setTimeout as wait } from "node:timers/promises";

const chromePath = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find(path => path && existsSync(path));

if (!chromePath) throw new Error("Chrome not found; set CHROME_PATH to run this check.");

const profile = mkdtempSync(join(tmpdir(), "gguf-explorer-"));
const chrome = spawn(chromePath, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--remote-debugging-pipe",
  `--user-data-dir=${profile}`,
], { stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"] });

let id = 0, buffer = "";
const pending = new Map(), errors = [], animations = [];

chrome.stdio[4].setEncoding("utf8");
chrome.stdio[4].on("data", chunk => {
  buffer += chunk;
  const messages = buffer.split("\0");
  buffer = messages.pop();
  messages.filter(Boolean).forEach(raw => {
    const message = JSON.parse(raw);
    if (message.id) {
      const call = pending.get(message.id);
      if (!call) return;
      pending.delete(message.id);
      message.error ? call.reject(new Error(message.error.message)) : call.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") {
      errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
    } else if (message.method === "Animation.animationStarted") {
      animations.push(message.params.animation.name);
    }
  });
});

function send(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const callId = ++id;
    pending.set(callId, { resolve, reject });
    chrome.stdio[3].write(JSON.stringify({ id: callId, method, params, sessionId }) + "\0");
  });
}

async function run() {
  const page = pathToFileURL(fileURLToPath(new URL("../index.html", import.meta.url))).href;
  const { targetId } = await send("Target.createTarget", { url: page });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const command = (method, params) => send(method, params, sessionId);
  const value = async expression => {
    const response = await command("Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
    return response.result.value;
  };

  await command("Runtime.enable");
  await command("Animation.enable");
  for (let tries = 0; tries < 40; tries++) {
    if (await value("document.readyState === 'complete'")) break;
    await wait(50);
  }

  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 4);
  await value("document.querySelectorAll('#xstack .xseg')[0].click(); true");
  await wait(400);
  assert.ok(animations.some(name => name.includes("group-anim-x-drill")));
  assert.deepEqual(await value(`({
    cards: document.querySelectorAll('#xstack .xseg').length,
    descriptions: document.querySelectorAll('#xstack .xd').length,
    buttons: document.querySelectorAll('#xstack button.xseg').length,
    focus: document.activeElement.className
  })`), { cards: 4, descriptions: 4, buttons: 4, focus: "xback" });

  await value("document.querySelector('#xstack .xseg').click(); true");
  assert.equal(await value("document.querySelector('.xtitle').textContent"), "magic");
  assert.ok((await value("document.querySelector('.xbody').textContent")).includes("G, G, U, F"));

  await value("document.querySelector('.xback').click(); true");
  await wait(400);
  assert.equal(await value("document.querySelector('.xcrumb').textContent"), "the whole file");
  await value("history.forward(); true");
  await wait(400);
  assert.ok((await value("document.querySelector('.xcrumb').textContent")).includes("header"));
  await value("history.back(); true");
  await wait(400);

  for (const index of [1, 2, 3]) {
    await value(`document.querySelectorAll('#xstack .xseg')[${index}].click(); true`);
    await wait(400);
    const counts = await value(`({
      cards: document.querySelectorAll('#xstack .xseg').length,
      descriptions: [...document.querySelectorAll('#xstack .xd')].filter(e => e.textContent.trim()).length
    })`);
    assert.equal(counts.descriptions, counts.cards);
    await value("history.back(); true");
    await wait(400);
  }

  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(400);
  await value("var s=document.querySelector('#modelselect');s.value='12b';s.dispatchEvent(new Event('change'));true");
  await value("history.back(); true");
  await wait(100);
  assert.equal(await value("document.querySelector('#modelselect').value"), "k3");
  await value("history.forward(); true");
  await wait(100);
  assert.equal(await value("document.querySelector('#modelselect').value"), "12b");

  await command("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }],
  });
  const drillCount = animations.filter(name => name.includes("group-anim-x-drill")).length;
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(50);
  assert.equal(animations.filter(name => name.includes("group-anim-x-drill")).length, drillCount);
  assert.deepEqual(errors, []);
}

try {
  await run();
  console.log("explorer interaction check passed");
} finally {
  if (chrome.exitCode === null) {
    chrome.kill();
    await once(chrome, "exit");
  }
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}
