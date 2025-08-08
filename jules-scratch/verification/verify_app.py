import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Get the absolute path to the index.html file
        html_file_path = os.path.abspath('index.html')

        await page.goto(f'file://{html_file_path}')
        await page.screenshot(path='jules-scratch/verification/01_initial_load.png')

        # Click on Semester 2
        await page.click('a[data-src="sem2.md"]')
        # Wait for the content to be loaded
        await page.wait_for_selector('h1:has-text("Semester 2")')
        await page.screenshot(path='jules-scratch/verification/02_semester2.png')

        # Click on Other Resources
        await page.click('a[data-src="other.md"]')
        # Wait for the content to be loaded
        await page.wait_for_selector('h2:has-text("Courses")')
        await page.screenshot(path='jules-scratch/verification/03_other_resources.png')

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
