"""Read-only fixture tools. No host filesystem, network, eval, exec, or gold access."""
import ast,operator,math,itertools,json

class ToolError(ValueError):pass
class Calculator:
 def __init__(self):self.remaining=100000
 def bound(self,v):
  if isinstance(v,int) and v.bit_length()>2048:raise ToolError('integer too large')
  if isinstance(v,(list,tuple,dict,str)) and len(v)>10000:raise ToolError('result too large')
  return v
 def call(self,name,args):
  if name=='range':
   v=range(*args)
   if len(v)>1000:raise ToolError('range limit')
   return list(v)
  if name=='product':
   if math.prod(len(x) for x in args)>10000:raise ToolError('product limit')
   return list(itertools.product(*args))
  if name=='zip':return list(zip(*args))
  funcs={'sum':sum,'min':min,'max':max,'len':len,'abs':abs,'sorted':sorted,'gcd':math.gcd,'lcm':math.lcm,'comb':math.comb}
  if name not in funcs:raise ToolError('unknown function')
  if name=='comb' and (args[0]>1000 or args[1]>1000):raise ToolError('comb limit')
  return self.bound(funcs[name](*args))
 def node(self,n,env):
  self.remaining-=1
  if self.remaining<0:raise ToolError('operation limit')
  if isinstance(n,ast.Constant):
   if type(n.value) not in [int,float,str,bool,type(None)]:raise ToolError('literal type')
   return self.bound(n.value)
  if isinstance(n,ast.Name):
   if n.id not in env:raise ToolError('unknown variable')
   return env[n.id]
  if isinstance(n,(ast.List,ast.Tuple)):return self.bound([self.node(x,env) for x in n.elts])
  if isinstance(n,ast.Dict):return self.bound({self.node(k,env):self.node(v,env) for k,v in zip(n.keys,n.values)})
  if isinstance(n,ast.UnaryOp):
   fn={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_}.get(type(n.op))
   if not fn:raise ToolError('unary operator')
   return self.bound(fn(self.node(n.operand,env)))
  if isinstance(n,ast.BinOp):
   a,b=self.node(n.left,env),self.node(n.right,env)
   if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):raise ToolError('numeric operands required')
   if isinstance(n.op,ast.Pow) and (abs(b)>12 or abs(a)>10**20):raise ToolError('power limit')
   fn={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}.get(type(n.op))
   if not fn:raise ToolError('binary operator')
   return self.bound(fn(a,b))
  if isinstance(n,ast.Compare):
   a=self.node(n.left,env)
   for op,bnode in zip(n.ops,n.comparators):
    b=self.node(bnode,env);fn={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b,ast.NotIn:lambda a,b:a not in b}.get(type(op))
    if not fn:raise ToolError('comparison')
    if not fn(a,b):return False
    a=b
   return True
  if isinstance(n,ast.BoolOp):
   for x in n.values:
    value=self.node(x,env)
    if isinstance(n.op,ast.And) and not value:return value
    if isinstance(n.op,ast.Or) and value:return value
   return value
  if isinstance(n,ast.IfExp):return self.node(n.body if self.node(n.test,env) else n.orelse,env)
  if isinstance(n,ast.Subscript):
   value=self.node(n.value,env)
   if not isinstance(value,(list,dict,str,tuple)):raise ToolError('subscript type')
   if isinstance(n.slice,ast.Slice):
    sl=slice(*(self.node(x,env) if x is not None else None for x in [n.slice.lower,n.slice.upper,n.slice.step]));return value[sl]
   return value[self.node(n.slice,env)]
  if isinstance(n,ast.Call):
   if not isinstance(n.func,ast.Name) or n.keywords:raise ToolError('named safe functions only')
   return self.call(n.func.id,[self.node(x,env) for x in n.args])
  if isinstance(n,(ast.ListComp,ast.GeneratorExp)):
   out=[]
   def bind(target,v,e):
    if isinstance(target,ast.Name):e[target.id]=v
    elif isinstance(target,(ast.Tuple,ast.List)) and len(target.elts)==len(v):
     for t,x in zip(target.elts,v):bind(t,x,e)
    else:raise ToolError('comprehension target')
   def visit(i,e):
    if i==len(n.generators):out.append(self.node(n.elt,e));self.bound(out);return
    gen=n.generators[i]
    if gen.is_async:raise ToolError('async unsupported')
    for v in self.node(gen.iter,e):
     ee=dict(e);bind(gen.target,v,ee)
     if all(self.node(x,ee) for x in gen.ifs):visit(i+1,ee)
   visit(0,env);return out
  raise ToolError('unsupported expression')
 def run(self,expression):
  if not isinstance(expression,str) or len(expression)>12000:raise ToolError('expression length')
  tree=ast.parse(expression,mode='eval')
  if len(list(ast.walk(tree)))>1000:raise ToolError('expression complexity')
  return self.bound(self.node(tree.body,{}))

TOOLS={
 'read':{'ref':'document reference; start at entry','description':'Read one document. For tables returns schema and row count; use query for rows.'},
 'search':{'text':'literal case-insensitive text','offset':'default 0','description':'Search document contents/titles, eight hits per page. Returns references and short previews.'},
 'query':{'ref':'table reference','where':'optional object of exact-match column filters','offset':'default 0','limit':'1..100, default 20','description':'Read paginated table rows; returns next_offset when more remain.'},
 'calculate':{'expression':'bounded Python-style numeric expression','description':'Arithmetic, comparisons, lists/dicts, comprehensions, indexing, sum/min/max/len/abs/sorted/gcd/lcm/comb/range/product/zip. No imports, attributes, assignment, host execution or filesystem. Up to 100000 evaluation steps.'}
}
class Workspace:
 def __init__(self,public):
  # Deliberately accepts only the public workspace, never the complete task.
  self.docs=json.loads(json.dumps(public['documents']));self.log=[]
 def call(self,name,args):
  try:
   if not isinstance(args,dict):raise ToolError('arguments must be an object')
   if name=='read':
    d=self.docs[args['ref']]
    out={'title':d['title'],'kind':d['kind'],'content':d['content']} if d['kind']!='table' else {'title':d['title'],'kind':'table','rows':len(d['rows']),'columns':list(d['rows'][0]) if d['rows'] else []}
   elif name=='search':
    text=args['text'].casefold();offset=args.get('offset',0)
    if not isinstance(offset,int) or offset<0:raise ToolError('offset')
    hits=[{'ref':ref,'title':d['title'],'preview':json.dumps(d.get('content',{}),ensure_ascii=False)[:120]} for ref,d in self.docs.items() if text in json.dumps(d,ensure_ascii=False).casefold()]
    out={'hits':hits[offset:offset+8],'next_offset':offset+8 if offset+8<len(hits) else None}
   elif name=='query':
    d=self.docs[args['ref']]
    if d['kind']!='table':raise ToolError('not a table')
    where=args.get('where',{});offset=args.get('offset',0);limit=args.get('limit',20)
    if not isinstance(where,dict) or type(offset)!=int or offset<0 or type(limit)!=int or not 1<=limit<=100:raise ToolError('invalid query')
    rows=[r for r in d['rows'] if all(k in r and r[k]==v for k,v in where.items())]
    out={'rows':rows[offset:offset+limit],'next_offset':offset+limit if offset+limit<len(rows) else None}
   elif name=='calculate':out={'result':Calculator().run(args['expression'])}
   else:raise ToolError('unknown tool')
   encoded=json.dumps(out,allow_nan=False)
   if len(encoded)>100000:raise ToolError('output limit')
  except (KeyError,ValueError,TypeError,IndexError,OverflowError,ZeroDivisionError,RecursionError,SyntaxError) as e:out={'error':str(e)[:250]}
  self.log.append({'tool':name,'arguments':args,'result':out});return out

def grade(expected,actual):
 # JSON field order irrelevant; booleans must not equal integers accidentally.
 def equal(a,b):
  if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  if type(a) in (int,float) and type(b) in (int,float):return a==b
  return type(a)==type(b) and a==b
 return int(equal(expected,actual))
