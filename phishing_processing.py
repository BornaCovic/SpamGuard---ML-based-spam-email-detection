import mailbox
import os
import pickle
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup as BS


def decoder(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset()
    try:
        return payload.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def extract_body(messg) -> str:
    parts = []
    for part in messg.walk():
        try:
            content_type = part.get_content_type()
            if content_type == "text/plain" and not part.is_attachment():
                parts.append(decoder(part))
            elif content_type == "text/html" and not part.is_attachment():
                html = decoder(part)
                soup = BS(html, "html.parser")
                parts.append(soup.get_text())
        except (UnicodeDecodeError, LookupError):
            continue
    return "\n".join(parts)


mbox_path = r"phishing-2025"
pickle_path = "data.pickle"

phishing_email_contents = []
for msg in mailbox.mbox(mbox_path):
    parsed = BytesParser(policy=policy.default).parsebytes(msg.as_bytes())
    body = extract_body(parsed)
    if body.strip() != "":
        phishing_email_contents.append(body.strip().lower())

if not os.path.exists(pickle_path):
    raise FileNotFoundError(
        f"{pickle_path} not found. Run data_processing.py first to create it."
    )
with open(pickle_path, "rb") as f:
    data_dict = pickle.load(f)

data_dict["spam"].extend(phishing_email_contents)
data_dict["spam"] = list(dict.fromkeys(data_dict["spam"]))

with open(pickle_path, "wb") as f:
    pickle.dump(data_dict, f)

print(f"Phishing emails parsed: {len(phishing_email_contents)}")
print(f"Total spam after merge (deduplicated): {len(data_dict['spam'])}")
print(f"Total ham: {len(data_dict['ham'])}")
