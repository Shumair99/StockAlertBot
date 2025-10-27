import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover
    async_playwright = None

load_dotenv()

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

OUT_OF_STOCK_INDICATORS = [
    "out of stock",
    "sold out",
    "unavailable",
    "currently unavailable",
    "coming soon",
    "notify me when available",
    "temporarily out of stock",
    "not available",
]


@dataclass
class Item:
    name: str
    url: str
    category: str = "ETB"
    last_in_stock: Optional[bool] = None
    last_change_ts: float = field(default_factory=lambda: 0.0)


DEFAULT_ITEMS: List[Item] = [
    Item(
        name: "Limited Release Booster Box",
        url: "https://store.example.com/product/booster-box",
        category: "Trading Cards",
    ),   
    Item(
        name="Collector's Elite Trainer Set",
        url="https://store.example.com/product/elite-trainer-set",
        category="Trading Cards",
    ),
    Item(
        name="Premium Figure Collection",
        url="https://store.example.com/product/premium-figure",
        category="Figures",
    ),
    Item(
        name="Exclusive Elite Trainer Set",
        url="https://store.example.com/product/exclusive-elite-trainer-set",
        category="Trading Cards",
    ),
    Item(
        name="Booster Bundle (8 packs)",
        url="https://store.example.com/product/booster-bundle",
        category="Trading Cards",
    ),
]


def _split_env_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[;,]\s*|\s+", raw)
    return [p for p in (s.strip() for s in parts) if p]


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_proxy_pool() -> List[Optional[str]]:
    ordered_keys = sorted(
        (k for k in os.environ if re.fullmatch(r"HTTP_PROXY_\d+", k)),
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )
    proxies: List[Optional[str]] = []
    for key in ordered_keys:
        val = os.getenv(key)
        if val:
            proxies.append(val)

    proxies.extend(_split_env_list(os.getenv("HTTP_PROXY_POOL")))
    proxies.extend(_split_env_list(os.getenv("HTTP_PROXIES")))

    single = os.getenv("HTTP_PROXY")
    if single:
        proxies.append(single)

    pw_single = os.getenv("PW_PROXY")
    if pw_single:
        proxies.append(pw_single)

    proxies.extend(_split_env_list(os.getenv("PW_PROXY_POOL")))

    unique: List[Optional[str]] = []
    for entry in proxies:
        if entry and entry not in unique:
            unique.append(entry)

    if not unique:
        unique.append(None)
    return unique


class ProxyManager:
    def __init__(self, proxies: Sequence[Optional[str]], rotation_threshold: int):
        if rotation_threshold < 1:
            rotation_threshold = 1
        self._proxies = list(proxies) or [None]
        self._index = 0
        self._blocks = 0
        self._threshold = rotation_threshold

    @property
    def current(self) -> Optional[str]:
        return self._proxies[self._index]

    def record_success(self) -> None:
        self._blocks = 0

    def record_block(self) -> bool:
        self._blocks += 1
        if len(self._proxies) == 1:
            return False
        if self._blocks < self._threshold:
            return False
        self._blocks = 0
        self._index = (self._index + 1) % len(self._proxies)
        return True


class PlaywrightFetcher:
    def __init__(self, user_agent: str, accept_language: str, proxy_manager: ProxyManager):
        if async_playwright is None:
            raise RuntimeError("Playwright not installed")
        self.user_agent = user_agent
        self.accept_language = accept_language
        self.proxy_manager = proxy_manager
        self.cookie = os.getenv("PC_COOKIE")
        self.state_path = os.getenv("PW_STATE", "playwright_state.json")
        self.keep_open = os.getenv("PW_KEEP_OPEN", "0").lower() in {"1", "true", "yes"}
        self.block_media = os.getenv("PW_BLOCK_MEDIA", "1").lower() in {"1", "true", "yes"}
        self.block_third_party = os.getenv("PW_BLOCK_THIRD_PARTY", "1").lower() in {"1", "true", "yes"}
        allowed = os.getenv("PW_ALLOWED_HOSTS", "example.com")
        self.allowed_hosts = [h.strip().lower() for h in allowed.split(",") if h.strip()]
        self._p = None
        self._browser = None
        self._context = None
        self._active_proxy: Optional[str] = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._p:
            await self._p.stop()
        self._context = self._browser = self._p = None
        self._active_proxy = None

    async def _ensure(self) -> None:
        desired_proxy = self.proxy_manager.current
        if self._context and desired_proxy == self._active_proxy:
            return
        await self.close()
        self._p = await async_playwright().start()
        headless = os.getenv("PW_HEADLESS", "1").lower() not in {"0", "false", "no"}
        slow_mo_env = os.getenv("PW_SLOWMO", "0")
        slow_mo = int(slow_mo_env) if slow_mo_env.isdigit() else 0
        browser_args = [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
        ]
        proxy_cfg = _to_playwright_proxy(desired_proxy)
        channel = os.getenv("PW_CHANNEL") or None
        try:
            self._browser = await self._p.chromium.launch(
                headless=headless,
                args=browser_args,
                channel=channel,
                slow_mo=slow_mo or None,
                proxy=proxy_cfg,
            )
        except Exception:
            self._browser = await self._p.chromium.launch(
                headless=headless,
                args=browser_args,
                slow_mo=slow_mo or None,
                proxy=proxy_cfg,
            )
        ctx_kwargs = {
            "user_agent": self.user_agent,
            "locale": self.accept_language.split(",")[0].split(";")[0],
            "viewport": {"width": 1366, "height": 768},
        }
        if self.state_path and os.path.exists(self.state_path):
            ctx_kwargs["storage_state"] = self.state_path
        self._context = await self._browser.new_context(**ctx_kwargs)

        async def _route(route, request):
            if self.block_media and request.resource_type in {"image", "media", "font", "stylesheet"}:
                return await route.abort()
            if self.block_third_party:
                host = request.url.split("//", 1)[-1].split("/", 1)[0].lower()
                if host and not any(host.endswith(h) for h in self.allowed_hosts):
                    return await route.abort()
            return await route.continue_()

        try:
            await self._context.route("**/*", _route)
        except Exception:
            pass

        for script in (
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});",
            "Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});",
            "Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});",
        ):
            await self._context.add_init_script(script)

        await self._context.set_extra_http_headers({"Accept-Language": self.accept_language})
        if self.cookie:
            domain = os.getenv("PRIMARY_DOMAIN", "example.com")
            cookies = _parse_cookie_string(self.cookie, domain=f".{domain.lstrip('.')}")
            if cookies:
                await self._context.add_cookies(cookies)
        self._active_proxy = desired_proxy

    async def get_html(self, url: str, timeout_ms: int = 30000) -> Optional[str]:
        async with self._lock:
            await self._ensure()
            if self._context is None:
                return None
            page = await self._context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                await page.wait_for_timeout(1500)
                html = await page.content()
                attempts = 2
                while attempts and _looks_like_waf_block(html):
                    attempts -= 1
                    await page.wait_for_timeout(3000)
                    try:
                        await page.reload(wait_until="networkidle", timeout=timeout_ms)
                    except Exception:
                        pass
                    html = await page.content()
                if html and not _looks_like_waf_block(html):
                    try:
                        if self.state_path:
                            await self._context.storage_state(path=self.state_path)
                    except Exception:
                        pass
                return html
            finally:
                if self.keep_open:
                    try:
                        await page.wait_for_timeout(600000)
                    except Exception:
                        pass
                else:
                    await page.close()


def _to_playwright_proxy(proxy_url: Optional[str]) -> Optional[Dict[str, str]]:
    if not proxy_url:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(proxy_url)
        server = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        cfg: Dict[str, str] = {"server": server}
        if parsed.username:
            cfg["username"] = parsed.username
        if parsed.password:
            cfg["password"] = parsed.password
        return cfg
    except Exception:
        return {"server": proxy_url}


def _parse_cookie_string(cookie: str, domain: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name.lower() in {"path", "domain", "expires", "max-age", "secure", "httponly", "samesite"}:
            continue
        pairs.append({"name": name, "value": value.strip(), "domain": domain, "path": "/"})
    return pairs


async def send_discord_webhook(
    client: httpx.AsyncClient,
    webhook_url: str,
    title: str,
    description: str,
    url: Optional[str] = None,
    color: int = 0x2ECC71,
    mention_here: bool = False,
) -> None:
    payload: Dict[str, object] = {
        "content": "@here" if mention_here else None,
        "allowed_mentions": {"parse": ["everyone"] if mention_here else []},
        "embeds": [
            {
                "title": title,
                "description": description,
                "url": url,
                "color": color,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ],
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    resp = await client.post(webhook_url, json=payload)
    if resp.status_code >= 300:
        print(f"[webhook] Error {resp.status_code}: {resp.text}")


def _looks_like_waf_block(html: str) -> bool:
    text = html.lower()
    return (
        "pardon our interruption" in text
        or "incapsula" in text
        or "/_incapsula_resource" in text
        or ("reese" in text and "protection" in text)
    )


def _has_unavailable_text(soup: BeautifulSoup) -> bool:
    page_text = soup.get_text(" ", strip=True).lower()
    if any(ind in page_text for ind in OUT_OF_STOCK_INDICATORS):
        return True
    return bool(soup.find(string=lambda t: isinstance(t, str) and "UNAVAILABLE" in t.upper()))


def _button_candidates(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
    yield from soup.find_all("button")
    yield from soup.find_all("a", attrs={"role": "button"})
    yield from soup.select("a.btn, a[class*=btn]")
    yield from soup.select("input[type=submit], input[type=button]")


def _element_matches_text(elem: BeautifulSoup, needle: str) -> bool:
    target = needle.lower()
    text = (elem.get_text(" ", strip=True) or "").lower()
    if target in text:
        return True
    value = str(elem.attrs.get("value") or "").lower()
    if target in value:
        return True
    for attr in ("aria-label", "title", "data-gtm", "data-analytics-title"):
        val = elem.attrs.get(attr)
        if isinstance(val, list):
            if any(target in str(v).lower() for v in val):
                return True
        elif isinstance(val, str) and target in val.lower():
            return True
    return False


def _element_disabled(elem: BeautifulSoup) -> bool:
    if "disabled" in elem.attrs:
        return True
    aria = elem.attrs.get("aria-disabled")
    if isinstance(aria, str) and aria.lower() in {"true", "1"}:
        return True
    classes = elem.attrs.get("class")
    if isinstance(classes, list):
        return any("disabled" in c.lower() for c in classes)
    if isinstance(classes, str):
        return "disabled" in classes.lower()
    return False


def _first_add_to_cart_button(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    labels = ["add to cart", "add to bag", "add to basket", "buy now"]
    for elem in _button_candidates(soup):
        if any(_element_matches_text(elem, label) for label in labels):
            if not _element_disabled(elem):
                return elem
    return None


def _schema_org_availability(soup: BeautifulSoup) -> Optional[bool]:
    for sc in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(sc.string or "{}")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for node in items:
            offers = node.get("offers") if isinstance(node, dict) else None
            if not offers:
                continue
            offer_list = offers if isinstance(offers, list) else [offers]
            for offer in offer_list:
                availability = str(
                    offer.get("availability")
                    or offer.get("itemAvailability")
                    or ""
                ).lower()
                if "instock" in availability:
                    return True
                if "outofstock" in availability:
                    return False
    return None


def detect_in_stock_for_storefront(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    avail = _schema_org_availability(soup)
    if avail is not None:
        return avail
    if _has_unavailable_text(soup):
        return bool(_first_add_to_cart_button(soup))
    if _first_add_to_cart_button(soup):
        return True
    oos = next((elem for elem in _button_candidates(soup) if _element_matches_text(elem, "out of stock")), None)
    if oos:
        return False
    return False


async def check_once(
    http_client: httpx.AsyncClient,
    webhook_url: Optional[str],
    item: Item,
    mention_here: bool,
    proxy_manager: ProxyManager,
    pw_fetcher: PlaywrightFetcher,
) -> Tuple[bool, Optional[bool]]:
    html = await pw_fetcher.get_html(item.url)
    blocked = not html or _looks_like_waf_block(html or "")

    if blocked:
        rotated = proxy_manager.record_block()
        if rotated:
            new_proxy = proxy_manager.current or "direct connection"
            print(f"[proxy] rotating to {new_proxy}")
        else:
            print(f"[blocked] Unable to bypass anti-bot for {item.url}")
        return True, None

    proxy_manager.record_success()
    print(f"[playwright] fetched {item.url}")

    in_stock = detect_in_stock_for_storefront(html)

    transitioned = item.last_in_stock is None or in_stock != item.last_in_stock
    if transitioned and webhook_url:
        item.last_change_ts = time.time()
        title = f"IN STOCK: {item.name}" if in_stock else f"Out of stock: {item.name}"
        desc = "Add to cart appears enabled." if in_stock else "Page indicates unavailable/sold out."
        color = 0x2ECC71 if in_stock else 0xE74C3C
        try:
            await send_discord_webhook(
                http_client,
                webhook_url,
                title=title,
                description=desc,
                url=item.url,
                color=color,
                mention_here=mention_here and in_stock,
            )
        except Exception as exc:
            print(f"[webhook] Exception: {exc}")

    item.last_in_stock = in_stock
    status = "IN STOCK" if in_stock else "out"
    print(f"[{status}] {item.name} -> {item.url}")
    return True, in_stock


async def run_monitor(
    items: List[Item],
    webhook_url: Optional[str],
    interval: float,
    concurrency: int = 4,
    mention_here: bool = False,
    run_once: bool = False,
    backend: str = "auto",
) -> None:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    }
    cookie = os.getenv("PC_COOKIE")
    if cookie:
        headers["Cookie"] = cookie

    rotation_threshold = _int_from_env("PROXY_ROTATE_AFTER", 3)
    proxy_manager = ProxyManager(_load_proxy_pool(), rotation_threshold)
    current_proxy = proxy_manager.current or "direct connection"
    print(f"[proxy] starting with {current_proxy}")

    requested_backend = (backend or os.getenv("FETCH_BACKEND", "playwright")).lower()
    if requested_backend not in {"auto", "playwright"}:
        print(f"[init] Backend '{requested_backend}' is no longer supported; using Playwright.")
    if async_playwright is None:
        print("[init] Playwright is not installed. Run 'pip install playwright' and 'playwright install'.")
        return

    global_pw: Optional[PlaywrightFetcher] = None
    try:
        global_pw = PlaywrightFetcher(
            user_agent=headers["User-Agent"],
            accept_language=headers["Accept-Language"],
            proxy_manager=proxy_manager,
        )
        if os.getenv("PW_WARMUP", "0").lower() in {"1", "true", "yes"}:
            warm_url = os.getenv("WARMUP_URL") or os.getenv("BASE_URL") or "https://example.com/"
            try:
                await global_pw.get_html(warm_url)
                print("[init] Playwright warm-up done")
            except Exception as exc:
                print(f"[init] Playwright warm-up failed: {exc}")
    except Exception as exc:
        print(f"[init] Playwright unavailable: {exc}")
        return

    if global_pw is None:
        print("[init] Playwright is required but could not be initialized.")
        return

    async with httpx.AsyncClient(timeout=15) as http_client:
        try:
            async def _run_checks() -> None:
                await asyncio.gather(
                    *(
                        check_once(
                            http_client,
                            webhook_url,
                            item,
                            mention_here,
                            proxy_manager,
                            global_pw,
                        )
                        for item in items
                    )
                )

            if run_once:
                await _run_checks()
                return

            while True:
                start = time.time()
                await _run_checks()
                elapsed = time.time() - start
                sleep_for = max(0.0, interval - elapsed)
                await asyncio.sleep(sleep_for)
        finally:
            try:
                await global_pw.close()
            except Exception:
                pass


def load_items_from_file(path: str) -> List[Item]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[Item] = []
    for row in data:
        items.append(
            Item(
                name=row.get("name") or row.get("title") or "Item",
                url=row["url"],
                category=row.get("category", "ETB"),
            )
        )
    return items


def parse_args(argv: List[str]):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stock monitor for limited-release trading card products with Discord "
            "webhook alerts."
        )
    )
    parser.add_argument(
        "--webhook",
        default=os.environ.get("DISCORD_WEBHOOK_URL"),
        help="Discord webhook URL (or set DISCORD_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--items-file",
        help='Path to JSON list of items: [{"name": "...", "url": "...", "category": "..."}]',
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Polling interval in seconds (default 60)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel requests limit",
    )
    parser.add_argument(
        "--mention-here",
        action="store_true",
        help="Mention @here when item is in stock",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "playwright"],
        default=os.environ.get("FETCH_BACKEND", "playwright"),
        help="Fetch backend: auto (default) or playwright (same behaviour).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a Playwright warm-up navigation before checks",
    )
    return parser.parse_args(argv)


def build_items(ns) -> List[Item]:
    if ns.items_file:
        return load_items_from_file(ns.items_file)
    return list(DEFAULT_ITEMS)


def main(argv: List[str]) -> int:
    ns = parse_args(argv)
    items = build_items(ns)
    if not items:
        print("No items configured. Add URLs via --items-file.")
        return 2

    if not ns.webhook:
        print(
            "No Discord webhook set. Set --webhook or DISCORD_WEBHOOK_URL.\n"
            "The monitor will still run but alerts won't be sent."
        )

    print(
        f"Monitoring {len(items)} item(s) every {ns.interval:.0f}s with concurrency {ns.concurrency}."
    )
    print(f"Backend: {ns.backend}")
    for it in items:
        print(f" - {it.category}: {it.name} -> {it.url}")

    try:
        asyncio.run(
            run_monitor(
                items,
                webhook_url=ns.webhook,
                interval=ns.interval,
                concurrency=ns.concurrency,
                mention_here=ns.mention_here,
                run_once=ns.once,
                backend=ns.backend,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
