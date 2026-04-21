"""
This module is a streamlined version of the 'gdown' library (https://github.com/wkentaro/gdown).
It has been adapted to provide a consistent interface with `mlalib.utils.download_from_url`
by suppressing metadata prints and returning pathlib.Path object for use in downloading datasets
supported by mlalib from Google Drive.
"""

import email.utils
import os
from pathlib import Path
import re
import shutil
import sys
from textwrap import wrap, indent
import time
import urllib.parse
import warnings
from http.cookiejar import MozillaCookieJar

import bs4
import requests
import tqdm


class FileURLRetrievalError(Exception):
    pass


def _is_google_drive_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname in ["drive.google.com", "docs.google.com"]


def _parse_url(url, warning=True):
    """Parse URLs especially for Google Drive links.

    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = _is_google_drive_url(url=url)
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return is_gdrive, is_download_link

    file_id = None
    if "id" in query:
        file_ids = query["id"]
        if len(file_ids) == 1:
            file_id = file_ids[0]
    else:
        patterns = [
            r"^/file/d/(.*?)/(edit|view)$",
            r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$",
            r"^/document/d/(.*?)/(edit|htmlview|view)$",
            r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.groups()[0]
                break

    if warning and not is_download_link:
        warnings.warn(
            "You specified a Google Drive link that is not the correct link "
            "to download a file. You might want to try `--fuzzy` option "
            "or the following url: {url}".format(
                url="https://drive.google.com/uc?id={}".format(file_id)
            )
        )

    return file_id, is_download_link


CHUNK_SIZE = 512 * 1024  # 512KB
home = Path.home()


def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        soup = bs4.BeautifulSoup(line, features="html.parser")
        form = soup.select_one("#download-form")
        if form is not None:
            url = form["action"].replace("&amp;", "&")
            url_components = urllib.parse.urlsplit(url)
            query_params = urllib.parse.parse_qs(url_components.query)

            for param in form.find_all("input", attrs={"type": "hidden"}):
                query_params[param["name"]] = param["value"]
            query = urllib.parse.urlencode(query_params, doseq=True)
            url = urllib.parse.urlunsplit(url_components._replace(query=query))
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise FileURLRetrievalError(error)
    if not url:
        raise FileURLRetrievalError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
            "Check FAQ in https://github.com/wkentaro/gdown?tab=readme-ov-file#faq.",
        )
    return url


def _get_filename_from_response(response):
    content_disposition = urllib.parse.unquote(response.headers["Content-Disposition"])

    m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
    if m:
        filename = m.groups()[0]
        return filename.replace("/", "_").replace("\\", "_")

    m = re.search('attachment; filename="(.*?)"', content_disposition)
    if m:
        filename = m.groups()[0]
        return filename

    return None


def _get_modified_time_from_response(response):
    if "Last-Modified" not in response.headers:
        return None

    raw = response.headers["Last-Modified"]
    if raw is None:
        return None

    return email.utils.parsedate_to_datetime(raw)


def _get_session(proxy, use_cookies, user_agent, return_cookies_file=False):
    sess = requests.session()

    sess.headers.update({"User-Agent": user_agent})

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy, file=sys.stderr)

    # Load cookies if exists
    cookies_file = Path(home) / ".cache/gdown/cookies.txt"
    cookies_file.parent.mkdir(parents=True, exist_ok=True)
    if use_cookies and cookies_file.exists():
        cookie_jar = MozillaCookieJar(str(cookies_file))
        cookie_jar.load()
        sess.cookies.update(cookie_jar)

    if return_cookies_file:
        return sess, cookies_file
    else:
        return sess


def download_from_gdrive(
    url: str | None = None,
    root: str | Path | None = None,
    filename: str | Path | None = None,
    proxy: str | None = None,
    speed: float | None = None,
    use_cookies: bool = True,
    verify: bool | str = True,
    id: str | None = None,
    fuzzy: bool = False,
    resume: bool = False,
    format: str | None = None,
    user_agent: str | None = None,
    log_messages: dict | None = None,
):
    """Download file from URL.

    Args:
        url (str or None): URL. Google Drive URL is also supported.
        root (str, Path or None): Optional directory in which to save the file or
        current working directory if None. Defaults to None.
        filename (str, Path or None): Optional name for file.
        If None, the name is inferred from the URL. Defaults to None.
        proxy (str or None): Proxy.
        speed (float or None): Download byte size per second (e.g., 256KB/s = 256 * 1024).
        use_cookies (bool): Flag to use cookies. Defaults to True.
        verify (bool or string): Either a bool, in which case it controls whether the server's TLS
        certificate is verified, or a string, in which case it must be a path
        to a CA bundle to use. Defaults to True.
        id (str or None): Google Drive's file ID.
        fuzzy (bool): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool): Resume interrupted downloads while skipping completed ones.
        Defaults to False.
        format (str or None):
            Format of Google Docs, Spreadsheets and Slides. Default is:
                - Google Docs: 'docx'
                - Google Spreadsheet: 'xlsx'
                - Google Slides: 'pptx'
        user_agent (str or None):
            User-agent to use in the HTTP request.

    Returns:
        Path: Path to the downloaded file.
    """
    root = Path(root) if root else Path.cwd()

    if filename:
        path = root / filename
        if path.is_file():
            return path

    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")
    if id is not None:
        url = "https://drive.google.com/uc?id={id}".format(id=id)
    if user_agent is None:
        # We need to use different user agent for file download c.f., folder
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA: E501

    url_origin = url

    sess, cookies_file = _get_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        return_cookies_file=True,
    )

    gdrive_file_id, is_gdrive_download_link = _parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
        url_origin = url
        is_gdrive_download_link = True

    while True:
        res = sess.get(url, stream=True, verify=verify)

        if not (gdrive_file_id and is_gdrive_download_link):
            break

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = "https://drive.google.com/open?id={id}".format(id=gdrive_file_id)
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            m = re.search("<title>(.+)</title>", res.text)
            if m and m.groups()[0].endswith(" - Google Docs"):
                url = (
                    "https://docs.google.com/document/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="docx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Sheets"):
                url = (
                    "https://docs.google.com/spreadsheets/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="xlsx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Slides"):
                url = (
                    "https://docs.google.com/presentation/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="pptx" if format is None else format,
                    )
                )
                continue
        elif (
            "Content-Disposition" in res.headers
            and res.headers["Content-Disposition"].endswith("pptx")
            and format not in {None, "pptx"}
        ):
            url = (
                "https://docs.google.com/presentation/d/{id}/export"
                "?format={format}".format(
                    id=gdrive_file_id,
                    format="pptx" if format is None else format,
                )
            )
            continue

        if use_cookies:
            cookie_jar = MozillaCookieJar(cookies_file)
            for cookie in sess.cookies:
                cookie_jar.set_cookie(cookie)
            cookie_jar.save()

        if "Content-Disposition" in res.headers:
            # This is the file
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            message = (
                "Failed to retrieve file url:\n\n{}\n\n"
                "You may still be able to access the file from the browser:"
                "\n\n\t{}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(
                indent("\n".join(wrap(str(e))), prefix="\t"),
                url_origin,
            )
            raise FileURLRetrievalError(message)

    filename_from_url = None
    last_modified_time = None
    if gdrive_file_id and is_gdrive_download_link:
        filename_from_url = _get_filename_from_response(response=res) or Path(url).name
        last_modified_time = _get_modified_time_from_response(response=res)

    filename = filename or filename_from_url
    path = root / filename
    if path.is_file():
        return path

    root.mkdir(parents=True, exist_ok=True)

    existing_tmp_files = []
    for file in path.parent.iterdir():
        if file.name.startswith(path.name) and file.suffix == ".part":
            existing_tmp_files.append(file)
    if resume and existing_tmp_files:
        if len(existing_tmp_files) != 1:
            print(
                "There are multiple temporary files to resume:",
                file=sys.stderr,
            )
            print("\n")
            for file in existing_tmp_files:
                print("\t", file, file=sys.stderr)
            print("\n")
            print(
                "Please remove them except one to resume downloading.",
                file=sys.stderr,
            )
            return
        tmp_file = existing_tmp_files[0]
    else:
        resume = False
        # mkstemp is preferred, but does not work on Windows
        # https://github.com/wkentaro/gdown/issues/153
        tmp_file = path.with_suffix(path.suffix + ".part")
    f = tmp_file.open("ab")

    if tmp_file is not None and f.tell() != 0:
        start_size = f.tell()
        headers = {"Range": "bytes={}-".format(start_size)}
        res = sess.get(url, headers=headers, stream=True, verify=verify)
    else:
        start_size = 0

    if resume:
        print("Resume:", tmp_file, file=sys.stderr)

    try:
        total = res.headers.get("Content-Length")
        if total is not None:
            total = int(total) + start_size
        pbar = tqdm.tqdm(
            total=total,
            unit="B",
            initial=start_size,
            unit_scale=True,
            desc=path.name,
        )
        t_start = time.time()
        downloaded = 0
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = downloaded / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, path)
        if last_modified_time:
            mtime = last_modified_time.timestamp()
            os.utime(path, (mtime, mtime))
    finally:
        sess.close()

    return path
