import stanza
import json
import tqdm.cli as tqdm

class Node:
    def __init__(self, labels=[]):
        self.pos = ""
        self.children = []
        self.parent = None
        self.depth = 0
        
        self._leaf_labels = labels
    
    def __str__(self) -> str:
        return "{pos} childrens {chind_n} labels {labes}".format(self.pos, len(self.children), self.leaf_labels())
    
    def __eq__(self, __value: object) -> bool:
        return self.pos == __value.pos and self.depth == __value.depth and self._leaf_labels == __value._leaf_labels

    def __ne__(self, __value: object) -> bool:
        return not self.__eq__(__value)
    
    def visit_preorder(self, internal=None, terminal=None):
        if self.is_terminal():
            if terminal:
                terminal(self)
        else:
            if internal:
                internal(self)
        
        for child in self.children:
            child.visit_preorder(internal, terminal)
    
    def leaf_labels(self):
        return self._leaf_labels
    
    def renew_labels(self):
        if self.is_terminal():
            return self._leaf_labels
        else:
            self._leaf_labels = []
            
            for child in self.children:
                self._leaf_labels.extend(child.renew_labels())
        
        return self._leaf_labels
    
    def is_terminal(self):
        return len(self.children) == 0
    
    def set_parent(self, parent):
        self.parent = parent
        self.depth = parent.depth + 1
        
class ConstituencyTree:
    def __init__(self) :
        # constituency 모델: ptb3_bert, wsj_bert
        self.stanlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package={'constituency': 'ptb3_bert'})
        self.sentence_tree = None
        self.original_tree = None
        self.original_sentence = None
    
    def __str__(self) -> str:
        indent = 0
        indented_tree = ""
        constituency_sentence = str(self.original_tree)
        for idx, char in enumerate(constituency_sentence):
            line = ""
            if char == "(":
                indent += 1
                indent_str = "\n" + (indent * "  ")  if indent != 1 else ""
                line = indent_str
                indented_tree += line
                
            elif char == ")":
                
                if constituency_sentence[idx-1] == ")":
                    indent_str = "\n" + (indent * "  ")
                    line = indent_str + char if indent != 1 else indent_str + char + "\n"
                    indented_tree += line
                else:
                    line = char if indent != 1 else char + "\n"
                    indented_tree += line 
                indent -= 1
                
            else:
                indented_tree += char
            
    def print_constituency(self):
        indent = 0
        constituency_sentence = str(self.original_tree)
        for idx, char in enumerate(constituency_sentence):
            line = ""
            if char == "(":
                indent += 1
                indent_str = "\n" + (indent * "  ")  if indent != 1 else ""
                print(indent_str + char ,end="")

            elif char == ")":
                
                if constituency_sentence[idx-1] == ")":
                    indent_str = "\n" + (indent * "  ")  if indent != 1 else "\n"
                    if indent != 1:
                        print(indent_str + char, end="")
                    else:
                        print(indent_str + char)
                else:
                    if indent != 1:
                        print(char, end="")
                    else:
                        print(char)
                indent -= 1
                
            else:
                print(char, end="")
            
            
    
    def sentence_to_tree(self, sentence: str):
        # stanford nlp가 제공하는 constituency Tree 객체는 자식 노드를 제거하는 기능이 없음
        # 수식 구조를 제거하기 위해서는 자식 노드를 제거하는 기능이 필요
        # 따라서 stanford nlp의 Tree를 trace하여 직접 구현한 Node 객체로 이식
        doc = self.stanlp(sentence)
        tree = doc.sentences[0].constituency
        self.original_tree = tree
        self.original_sentence = sentence
        
        self.sentence_tree = Node(tree.leaf_labels())
        self.sentence_tree.pos = "ROOT"
        
        self.tree_depth = tree.depth() - 1
        
        self.parent_node = [self.sentence_tree]
        
        def internal_func(node):
            if node.label != "ROOT":
                node_dict = Node(node.leaf_labels())
                node_dict.pos = node.label
                node_dict.set_parent(self.parent_node[-1])
                
                self.parent_node[-1].children.append(node_dict)
                
                self.parent_node.append(node_dict)
        
        def preterminal_func(node):
            node_dict = Node(node.leaf_labels())
            node_dict.pos = node.label
            node_dict.set_parent(self.parent_node[-1])
            
            self.parent_node[-1].children.append(node_dict)
            
            cond = False
            while not cond and len(self.parent_node) > 0:
                child_labels = [word for child in self.parent_node[-1].children for word in child._leaf_labels]
                if len(child_labels) == len(self.parent_node[-1].leaf_labels()):
                    self.parent_node.pop()
                else:
                    cond = True
        
        tree.visit_preorder(
            internal=internal_func,
            preterminal=preterminal_func
        )
        
        return self.sentence_tree