from collections import Counter
import re

nums = set()
for i in range(0, 10):
    nums.add(str(i))


def has_num(string):
    for ch in string:
        if ch in nums:
            return True
    return False


def get_text(filename):
    f = open(filename, "r")
    text = [line.replace("\n", "").lower() for line in f.readlines() if
            not has_num(line) and len(line.strip("\n")) != 0]
    return text


def count_words(filename):
    text = get_text(filename)
    c = Counter()
    for line in text:
        line = re.findall(r"[\w']+|[.,!?;]", line)
        for word in line:
            c[word] += 1
    del c["s"]
    return c




