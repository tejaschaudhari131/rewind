from pathlib import Path
import re, json, subprocess
base=Path(__file__).resolve().parent
repo=base/'repo'
sha='bc7cd2b04425220965bc732b8b80ce54da450375'
path=base/'AUDIT.md'
content=path.read_text(encoding='utf-8')
count=0
def cite(m):
    global count
    file,start,end=m.group(1),int(m.group(2)),int(m.group(3))
    lines=(repo/file).read_text(encoding='utf-8').splitlines()
    if not (1 <= start <= end <= len(lines)):
        raise ValueError((file,start,end,len(lines)))
    count+=1
    return f'[{file}:{start}–{end}](https://github.com/tejaschaudhari131/rewind/blob/{sha}/{file}#L{start}-L{end})'
content=re.sub(r'\{\{([^:}]+):(\d+)-(\d+)\}\}',cite,content)
assert '{{' not in content
path.write_text(content,encoding='utf-8')
results=json.loads((base/'reproduction-results.json').read_text())
errors={k:v for k,v in results.items() if isinstance(v,dict) and 'error' in v}
assert not errors,errors
print(json.dumps({'citations_validated':count,'probe_groups':len(results),'report_words':len(content.split()),'results_errors':errors}))
