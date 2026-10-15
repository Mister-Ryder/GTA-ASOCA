"""User-authorized new public repository; existing Git auth stays in memory."""
import argparse,json,os,subprocess,urllib.request,urllib.error
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['check','create'],required=True)
p.add_argument('--name',default='GTA-ASOCA');p.add_argument('--receipt',required=True)
p.add_argument('--direct',action='store_true')
a=p.parse_args()
if not a.name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid repository name')
env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
process=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\npath=Mister-Ryder/GTAirway.git\n\n',
                       capture_output=True,text=True,env=env,timeout=45)
if process.returncode:raise RuntimeError('Existing Git authentication unavailable; no credential output was printed')
credential=dict(line.split('=',1) for line in process.stdout.splitlines() if '=' in line)
token=credential.get('password')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.direct else urllib.request.build_opener()
if not token:raise RuntimeError('Existing Git auth returned no usable credential')
def api(method,path,data=None):
 req=urllib.request.Request('https://api.github.com'+path,method=method,
  data=json.dumps(data).encode() if data is not None else None,
  headers={'Accept':'application/vnd.github+json','Authorization':'Bearer '+token,
           'X-GitHub-Api-Version':'2026-03-10','User-Agent':'GTA-ASOCA-source-release','Content-Type':'application/json'})
 try:
  with opener.open(req,timeout=45) as resp:return resp.status,json.loads(resp.read())
 except urllib.error.HTTPError as e:
  try:message=json.loads(e.read()).get('message','HTTP failure')
  except Exception:message='HTTP failure'
  return e.code,{'message':message}
code,profile=api('GET','/user')
if code!=200 or profile.get('login','').lower()!='mister-ryder':
 raise RuntimeError('GitHub owner/authentication check failed; status='+str(code))
code,repo=api('GET','/repos/Mister-Ryder/'+a.name)
if a.mode=='check':
 print(json.dumps({'owner':profile['login'],'owner_id':profile['id'],'repository_name':a.name,
                   'repository_exists':code==200,'repository_lookup_status':code,'authenticated':True}))
else:
 receipt=Path(a.receipt)
 if receipt.exists():raise RuntimeError('Creation has a durable receipt; verify it instead of creating again')
 if code!=404:raise RuntimeError('New repository name is not unused or cannot be verified; lookup status='+str(code))
 code,repo=api('POST','/user/repos',{'name':a.name,'private':False,'auto_init':False,
   'description':'Development implementation of GTA prior-guided multitask coronary segmentation on ASOCA; frozen reproducible source and training diagnostics.',
   'has_projects':False,'has_wiki':False})
 if code!=201:raise RuntimeError('GitHub repository creation failed; status='+str(code)+'; '+repo.get('message','HTTP failure'))
 if repo.get('private') is not False or repo.get('owner',{}).get('login','').lower()!='mister-ryder':raise RuntimeError('Created repository visibility/owner differs')
 value={k:repo.get(k) for k in ['id','full_name','html_url','clone_url','private','visibility','default_branch']}
 value.update(owner_id=profile['id'],user_authorized_new_public_source_repository=True)
 receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(value))
token=None;credential.clear()
