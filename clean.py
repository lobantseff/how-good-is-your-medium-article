import re as _re
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = _re.sub('\s\W', ' ', s)
    s = _re.sub('\W,\s', ' ', s)
    s = _re.sub(r'[^\w]', ' ', s)
    s = _re.sub('\d+', '', s)
    s = _re.sub('\s+', ' ', s)
    s = _re.sub('[!@#$_]', '', s)
    s = s.replace(',', '')
    s = s.replace('[\w*', ' ')
    s = _re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=_re.MULTILINE)
    s = _re.sub(r'\<a href', ' ', s)
    s = _re.sub(r'&amp;', '', s)
    s = _re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = _re.sub(r'[^\x00-\x7f]', r'', s)
    s = _re.sub(r'<br />', ' ', s)
    s = _re.sub(r'\'', ' ', s)
    return s