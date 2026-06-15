const puppeteer = require('puppeteer-core');

const slides = Array.from({length: 13}, (_, i) => {
  const n = String(i + 1).padStart(2, '0');
  return {
    html: `/Users/I558118/Documents/Projects/Whisper-Ng/slides-light/slide${n}.html`,
    png:  `/Users/I558118/Documents/Projects/Whisper-Ng/slides-light/slide${n}.png`
  };
});

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security'],
    headless: 'new'
  });

  for (const slide of slides) {
    console.log(`Rendering ${slide.html} ...`);
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 2 });
    await page.goto(`file://${slide.html}`, { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1200));
    await page.screenshot({ path: slide.png, type: 'png' });
    await page.close();
    console.log(`  → saved ${slide.png}`);
  }

  await browser.close();
  console.log('All light slides rendered.');
})();
