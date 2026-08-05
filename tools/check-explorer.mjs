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
const pending = new Map(), errors = [];

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
    if (response.exceptionDetails) throw new Error(`${response.exceptionDetails.text}: ${expression}`);
    return response.result.value;
  };
  const shellHorizontalFrames = `(() => {
    const frames = document.querySelector('.xmotion-shell').getAnimations()[0].effect.getKeyframes();
    return {
      leftPositions: new Set(frames.map(frame => frame.left)).size,
      widths: new Set(frames.map(frame => frame.width)).size
    };
  })()`;

  await command("Runtime.enable");
  for (let tries = 0; tries < 40; tries++) {
    if (await value("document.readyState === 'complete'")) break;
    await wait(50);
  }

  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 4);
  await value("document.documentElement.style.scrollBehavior='auto';document.querySelector('.explorer').scrollIntoView({block:'center'});true");
  const scrollTop = await value("window.scrollY");
  const firstHeight = await value("document.querySelector('#xstack .xseg').getBoundingClientRect().height");
  await value("document.querySelectorAll('#xstack .xseg')[0].click(); true");
  await wait(50);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage .xold').length"), 1);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage .xnew').length"), 1);
  assert.equal(await value("document.querySelectorAll('.xmotion-shell').length"), 1);
  assert.equal(await value("document.querySelector('.xmotion-shell').children.length"), 0);
  assert.deepEqual(await value(shellHorizontalFrames), { leftPositions: 1, widths: 1 });
  assert.ok(await value(`Math.abs(window.scrollY - ${scrollTop}) < 1`), "forward drill should not move the page");
  assert.ok(await value(`document.querySelector('.xmotion-shell').getBoundingClientRect().height > ${firstHeight}`),
            "forward shell should expand");
  await wait(100);
  assert.ok(await value(`(() => {
    const old = parseFloat(getComputedStyle(document.querySelector('.xold')).opacity);
    const fresh = parseFloat(getComputedStyle(document.querySelector('.xnew')).opacity);
    return old > 0 && old < 1 && fresh === 0;
  })()`), "old labels should fade before new labels appear");
  await wait(100);
  assert.ok(await value(`(() => {
    const old = parseFloat(getComputedStyle(document.querySelector('.xold')).opacity);
    const fresh = parseFloat(getComputedStyle(document.querySelector('.xnew')).opacity);
    const shell = parseFloat(getComputedStyle(document.querySelector('.xmotion-shell')).opacity);
    return old === 0 && fresh > 0 && fresh < 1 && shell > 0 && shell < 1;
  })()`), "new sections should cross-dissolve with the expanded shell");
  await wait(200);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage').length"), 0);
  assert.equal(await value("document.querySelector('#xstack').style.visibility"), "");
  assert.ok(await value(`Math.abs(window.scrollY - ${scrollTop}) < 1`));
  assert.deepEqual(await value(`({
    cards: document.querySelectorAll('#xstack .xseg').length,
    descriptions: document.querySelectorAll('#xstack .xd').length,
    buttons: document.querySelectorAll('#xstack button.xseg').length,
    visible: [...document.querySelectorAll('#xstack .xseg')].every(e => e.scrollHeight <= e.clientHeight + 1),
    focus: document.activeElement.className
  })`), { cards: 4, descriptions: 4, buttons: 4, visible: true, focus: "xback" });

  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(50);
  assert.equal(await value("document.querySelectorAll('.xmotion-shell').length"), 1);
  await wait(400);
  assert.equal(await value("document.querySelectorAll('#xstack .terminal').length"), 1);
  assert.ok((await value("document.querySelector('.xcrumb').textContent")).includes("magic"));
  assert.equal(await value("document.querySelector('.xtitle').textContent"), "magic");
  assert.ok((await value("document.querySelector('.xbody').textContent")).includes("G, G, U, F"));
  assert.ok((await value("document.querySelector('.terminal .xd').textContent")).includes("G, G, U, F"));
  assert.equal(await value("document.querySelector('.terminal').scrollHeight <= document.querySelector('.terminal').clientHeight + 1"), true);

  const terminalHeight = await value("document.querySelector('#xstack').getBoundingClientRect().height");
  await value("document.querySelector('.xback').click(); true");
  await wait(50);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage .xold').length"), 1);
  assert.equal(await value("document.querySelectorAll('.xmotion-shell').length"), 1);
  assert.deepEqual(await value(shellHorizontalFrames), { leftPositions: 1, widths: 1 });
  assert.ok(await value(`document.querySelector('.xmotion-shell').getBoundingClientRect().height < ${terminalHeight}`),
            "back shell should contract");
  await wait(400);
  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 4);
  assert.ok((await value("document.querySelector('.xcrumb').textContent")).includes("header"));

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
  await wait(50);
  await value("history.back(); true");
  await wait(400);
  assert.equal(await value("document.querySelector('.xcrumb').textContent"), "the whole file");
  assert.equal(await value("document.querySelectorAll('.xmotion-stage').length"), 0);
  assert.equal(await value("document.querySelector('#xstack').style.visibility"), "");
  assert.equal(await value("document.querySelector('.xcol').classList.contains('moving')"), false);
  assert.equal(await value("document.documentElement.classList.contains('xmoving')"), false);

  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(400);
  await value("var s=document.querySelector('#modelselect');s.value='12b';s.dispatchEvent(new Event('change'));true");
  await value("history.back(); true");
  await wait(100);
  assert.equal(await value("document.querySelector('#modelselect').value"), "k3");
  await value("history.forward(); true");
  await wait(100);
  assert.equal(await value("document.querySelector('#modelselect').value"), "12b");

  await value("var s=document.querySelector('#modelselect');s.value='k3';s.dispatchEvent(new Event('change'));true");
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(50);
  await value("var s=document.querySelector('#modelselect');s.value='12b';s.dispatchEvent(new Event('change'));true");
  await wait(50);
  assert.equal(await value("document.querySelector('.xcrumb').textContent"), "the whole file");
  assert.equal(await value("document.querySelectorAll('.xmotion-stage').length"), 0);
  assert.equal(await value("document.querySelector('#xstack').style.visibility"), "");
  assert.equal(await value("document.querySelector('.xcol').classList.contains('moving')"), false);
  assert.equal(await value("document.documentElement.classList.contains('xmoving')"), false);

  await value("var s=document.querySelector('#modelselect');s.value='k3';s.dispatchEvent(new Event('change'));true");
  await value("document.querySelectorAll('#xstack .xseg')[3].click(); true");
  await wait(400);
  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 6);
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(400);
  assert.deepEqual(await value(`({
    cards: document.querySelectorAll('#xstack .xseg').length,
    descriptions: document.querySelectorAll('#xstack .xd').length,
    first: document.querySelector('#xstack .xn').textContent,
    firstSize: document.querySelector('#xstack .xv').textContent,
    last: document.querySelector('#xstack .xseg:last-child .xn').textContent,
    explanation: document.querySelector('.xbody').textContent
  })`), {
    cards: 14,
    descriptions: 14,
    first: "expert slots 000–063",
    firstSize: "103.32 GB",
    last: "expert slots 832–895",
    explanation: "896 numbered expert slots repeat through 92 model layers — 82,432 distinct expert modules in the file.Open a range to reach any slot exactly. Each slot contains gate, up, and down slices in every expert-bearing layer."
  });

  await value("document.querySelector('.xcol').scrollTop=180;true");
  const rangeScroll = await value("document.querySelector('.xcol').scrollTop");
  await value("document.querySelectorAll('#xstack .xseg')[4].click();true");
  await wait(400);
  assert.equal(await value("document.querySelector('.xcol').scrollTop"), 0);
  assert.equal(await value("document.querySelector('#xstack .xn').textContent"), "expert slots 256–263");
  assert.equal(await value("document.querySelector('#xstack .xv').textContent"), "12.91 GB");
  await value("document.querySelector('.xback').click(); true");
  await wait(400);
  assert.equal(await value("document.querySelector('.xcol').scrollTop"), rangeScroll);

  await value("document.querySelector('.xcol').scrollTop=0;document.querySelector('#xstack .xseg').click();true");
  await wait(400);
  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 8);
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(400);
  assert.equal(await value("document.querySelectorAll('#xstack .xseg').length"), 8);
  assert.equal(await value("document.querySelector('#xstack .xn').textContent"), "expert slot 000");
  assert.equal(await value("document.querySelector('#xstack .xv').textContent"), "1.61 GB");
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(400);
  assert.equal(await value("document.querySelector('.terminal .xn').textContent"), "expert slot 000");
  assert.ok((await value("document.querySelector('.terminal .xd').textContent")).includes("92 distinct expert modules"));
  await value("document.querySelector('.xcrumb button').click(); true");
  await wait(500);

  await command("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }],
  });
  assert.equal(await value("document.querySelector('.xcrumb').textContent"), "the whole file");
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(50);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage').length"), 0);
  assert.equal(await value("document.documentElement.classList.contains('xmoving')"), false);
  assert.equal(await value("document.querySelectorAll('#xstack .xd').length"), 4);
  await value("document.querySelector('#xstack .xseg').click(); true");
  await wait(50);
  assert.equal(await value("document.querySelectorAll('.xmotion-stage').length"), 0);
  assert.equal(await value("document.querySelectorAll('#xstack .terminal').length"), 1);

  await value("document.querySelector('.xcrumb button').click(); true");
  await wait(100);
  await command("Emulation.setDeviceMetricsOverride", {
    width: 390, height: 844, deviceScaleFactor: 1, mobile: true,
  });
  await value("document.querySelectorAll('#xstack .xseg')[3].click();true");
  await value("document.querySelector('#xstack .xseg').click();true");
  await wait(50);
  assert.deepEqual(await value(`({
    cards: document.querySelectorAll('#xstack .xseg').length,
    textFits: [...document.querySelectorAll('#xstack .xseg')].every(e => e.scrollHeight <= e.clientHeight + 1),
    noOverflow: document.querySelector('.xcol').scrollWidth <= document.querySelector('.xcol').clientWidth,
    headerGap: Math.round(document.querySelector('#xtop').getBoundingClientRect().top -
                          document.querySelector('.xcrumb').getBoundingClientRect().bottom)
  })`), { cards: 14, textFits: true, noOverflow: true, headerGap: 10 });

  await value("document.querySelector('.xcrumb button').click();true");
  await command("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "no-preference" }],
  });
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1000, height: 768, deviceScaleFactor: 1, mobile: false,
  });
  await value("document.querySelectorAll('#xstack .xseg')[3].click();true");
  await wait(400);
  await value("document.querySelector('#xstack .xseg').click();true");
  await wait(50);
  assert.deepEqual(await value(`({
    crumbFits: document.querySelector('.xcrumb').scrollHeight <= document.querySelector('.xcrumb').clientHeight,
    headerClear: document.querySelector('#xtop').getBoundingClientRect().top >=
                 document.querySelector('.xcrumb').getBoundingClientRect().bottom + 9
  })`), { crumbFits: true, headerClear: true });
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
