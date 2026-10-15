"""Verify anonymous public access and every published Git blob, without auth."""
import argparse,hashlib,json,subprocess,time,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
r=Path(a.root).resolve();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(path):
 for attempt in range(3):
  try:
   req=urllib.request.Request('https://api.github.com/repos/Mister-Ryder/GTA-ASOCA'+path,
       headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10','User-Agent':'GTA-ASOCA-public-source-verification'})
   with opener.open(req,timeout=30) as response:return json.loads(response.read())
  except urllib.error.URLError:
   if attempt==2:raise
   time.sleep(1)
meta=get('')
if meta.get('private') is not False or meta.get('visibility')!='public':raise RuntimeError('Repository is not anonymously public')
local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
ref=get('/git/ref/heads/main')['object']
if ref.get('sha')!=local:raise RuntimeError('Published branch differs from reviewed commit')
commit=get('/git/commits/'+local)
tree=get('/git/trees/'+commit['tree']['sha']+'?recursive=1')
if tree.get('truncated'):raise RuntimeError('Public source tree is truncated')
remote={entry['path']:entry for entry in tree['tree'] if entry['type']=='blob'}
expected=json.loads((r/'PUBLIC_SOURCE_PROVENANCE.json').read_text())['public_file_allowlist']+['PUBLIC_SOURCE_PROVENANCE.json']
if set(remote)!=set(expected):raise RuntimeError('Published source file membership differs')
for name,item in remote.items():
 data=(r/name).read_bytes();digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if item['sha']!=digest or item.get('size')!=len(data):raise RuntimeError('Published source byte identity differs: '+name)
value={'status':'verified_public','url':meta['html_url'],'private':False,'commit':local,'branch':'main',
       'source_file_count':len(remote),'all_published_blobs_match_reviewed_bytes':True,'anonymous_access_verified':True}
path=Path(a.receipt);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
