import antlr4
from java.JavaLexer import JavaLexer
from java.JavaParserModified import JavaParserModified
from queue import Queue

def nname(node):
  if "TerminalNodeImpl" in str(node.__class__):
    return (node.getText(), "T")
  else:
    return (str(node.__class__).split('.')[-1][:-9], "NT")

def getRuleAtNode(node):
  rule = []
  (name, typ) = nname(node)
  for i in range(0, node.getChildCount()):
    ch = node.getChild(i)
    rule.append(nname(ch)[0])
  return name + '-->' + '___'.join(rule)

def getProductions(code):
  stream = antlr4.InputStream(code)
  lexer = JavaLexer(stream)
  toks = antlr4.CommonTokenStream(lexer)
  parser = JavaParserModified(toks)

  # We are always passing methods
  tree = parser.memberDeclaration()

  # Run a transition based parser on it to generate the dataset
  st = []
  st.append(tree)

  rule_seq = []

  while(len(st) > 0):
    top = st.pop()
    (name, typ) = nname(top)
    if name == "ErrorN":
      return None # There is a parsing error
    if typ == "T": # Terminal
      pass
    else: # Non-terminal
      rule = getRuleAtNode(top)
      rule_seq.append(rule)
      # put the rule in to the buffer
      for i in range(top.getChildCount() - 1, -1, -1):
        st.append(top.getChild(i))

  # Ignore the first 6 production rules
  return rule_seq[6:]
