from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup as BS
import os
from collections import Counter
import pickle

def decoder(part) -> str:
    payload = part.get_payload(decode=True)
    charset = part.get_content_charset()
    try:
        return payload.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")
    
def read_message_body(eml_path: str) -> str:
    with open(eml_path, "rb") as f:
        messg = BytesParser(policy=policy.default).parse(f)

        parts = []
        for part in messg.walk():
            try:
                attachment_ = part.is_attachment()
                content_type = part.get_content_type()
                if part.get_content_type() == "text/plain" and not part.is_attachment():
                    parts.append(decoder(part))
                if part.get_content_type() == "text/html" and not part.is_attachment():
                    html = decoder(part)
                    soup = BS(html, "html.parser")
                    parts.append(soup.get_text())
            except (UnicodeDecodeError, LookupError) as e:
                continue
        return "\n".join(parts)
    
directory = r"spam_2"

spam_email_contents = []
for name in os.listdir(directory):
    file_path = os.path.join(directory, name)
    if os.path.isfile(file_path):
        parts = read_message_body(file_path)
        if parts != "":
            spam_email_contents.append(parts.strip())

directory = r"easy_ham_2"
ham_email_contents = []
for name in os.listdir(directory):
    file_path = os.path.join(directory, name)
    if os.path.isfile(file_path):
        parts = read_message_body(file_path)
        if parts != "":
            ham_email_contents.append(parts.strip())

spam_email_contents = list(dict.fromkeys(spam_email_contents))
ham_email_contents = list(dict.fromkeys(ham_email_contents))
with open('data.pickle', 'wb') as f:
    pickle.dump({"spam" : spam_email_contents, "ham" : ham_email_contents}, f)
