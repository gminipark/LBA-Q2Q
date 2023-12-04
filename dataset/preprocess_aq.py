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
    
    
    def remove_modifier(self, key_entity:str):
        
        def find_entity_node(node, key_entity, entity_node_list):
            
            if key_entity in node.leaf_labels():
                if node.is_terminal():
                    entity_node_list.append(node)
                    return entity_node_list
                else:
                    for child in node.children:
                        entity_node_list = find_entity_node(child, key_entity, entity_node_list)
                
            return entity_node_list
    
        def get_entity_root_node(node, key_entity):
            
            while node.pos in ["NP", "NN", "NNS", "NNP", "NNPS"]:
                 node = node.parent
                 
            return node
        
        
        # 주어진 key entity에 대한 수식어를 제거하는 함수
        cur_node = self.sentence_tree
        question_type = self.sentence_tree.children[0].pos
        if question_type == "SBARQ" or "SQ":
            cur_node = cur_node.children[0]
        entity_found = True
        removed_tokens = []
        entity_root_node = None
        
        
        entity_node_list = []
        
        for child in cur_node.children:
            find_entity_node(child, key_entity, entity_node_list)
       
        node_set_list = []
        
        for entity_node in entity_node_list:
            root_node = get_entity_root_node(entity_node, key_entity)
            
            node_set_list.append((entity_node, root_node))
            
        # if len(node_set_list) > 1:
        #     for (entity_node, entity_root_node) in node_set_list:
        #         print(entity_node.leaf_labels())
        #         print(entity_root_node.leaf_labels())
        #     exit()
        
        for (entity_node, entity_root_node) in node_set_list:
            while entity_node != entity_root_node:
                previous_node = entity_node
                #print(entity_node.leaf_labels())
                entity_node = entity_node.parent
                #print(entity_node.leaf_labels() == entity_root_node.leaf_labels())
              
                children_pos_list = [child.pos for child in entity_node.children]

                if previous_node.pos in ["NP", "NN", "NNS", "NNP", "NNPS"]:
                    removed_token = None
                    if "JJ" in children_pos_list:
                        # 형용사 수식 : ex) silver car
                        entity_found = True
                                
                        jj_idx = children_pos_list.index("JJ")
                        removed_token = entity_node.children[jj_idx].leaf_labels()
                        removed_tokens.append(removed_token)
                        entity_node.children.remove(entity_node.children[jj_idx])
                        children_pos_list = [child.pos for child in entity_node.children]
                                
                    if "ADJP" in children_pos_list:
                        # 형용사구 : ex) Brown or yellow car
                        entity_found = True
                        
                        adjp_idx = children_pos_list.index("ADJP")
                        removed_token = entity_node.children[adjp_idx].leaf_labels()
                        removed_tokens.append(removed_token)
                        entity_node.children.remove(entity_node.children[adjp_idx])
                        children_pos_list = [child.pos for child in entity_node.children]
                
                    if "SBAR" in children_pos_list:
                        # that~절 수식 구조
                        sbar_idx = children_pos_list.index("SBAR")
                        
                        if key_entity not in entity_node.children[sbar_idx].leaf_labels():
                            removed_tokens.append(entity_node.children[sbar_idx].leaf_labels())
                            entity_node.children.remove(entity_node.children[sbar_idx])
                            children_pos_list = [child.pos for child in entity_node.children]
                    
                    #if entity_node.pos == "NP" in children_pos_list:
                        
                        
                        # 현재 노드가 명사절(NP)일 때
                        
                        # if question_type == "SBARQ" or \
                        #     (parent_child_pos_list and "VB" not in parent_child_pos_list[cur_child_idx - 1]):
                        #     # pos를 만족한 상태에서 
                        #     # 질문 종류를 만족하거나
                        #     # 전치사구가 의문문의 일부가 아니거나
                    
                            # (the man in the street)와 같이 수식된 경우
                            
                    if "PP" in children_pos_list:
                       
                        if entity_node.pos != "SQ" and entity_node.parent.pos != "SQ": # and is_complement:
                            #self.print_constituency()
                            
                            pp_idx = children_pos_list.index("PP")
                            removed_token = entity_node.children[pp_idx].leaf_labels()
                            removed_tokens.append(removed_token)
                            entity_node.children.remove(entity_node.children[pp_idx])
                            children_pos_list = [child.pos for child in entity_node.children]
                        
                        elif entity_node.parent.pos == "SQ":
                            align_pos_list = [align_node.pos if align_node != entity_node else "" for align_node in entity_node.parent.children]
                            
                            is_complement = True if "VP" in align_pos_list  else False
                            
                            if is_complement:
                                #print("is complement")
                                #self.print_constituency()
                                pp_idx = children_pos_list.index("PP")
                                removed_token = entity_node.children[pp_idx].leaf_labels()
                                removed_tokens.append(removed_token)
                                entity_node.children.remove(entity_node.children[pp_idx])
                                children_pos_list = [child.pos for child in entity_node.children]
                            
                        else:
                            is_complement = True if "VP" in children_pos_list  else False
                            
                            if is_complement:
                                #print("is complement")
                                #self.print_constituency()
                                pp_idx = children_pos_list.index("PP")
                                removed_token = entity_node.children[pp_idx].leaf_labels()
                                removed_tokens.append(removed_token)
                                entity_node.children.remove(entity_node.children[pp_idx])
                                children_pos_list = [child.pos for child in entity_node.children]
                                
                        
                        align_pos_list = [align_node.pos if align_node != entity_node else "" for align_node in entity_node.parent.children]
                        # num_vp = align_pos_list.count("VP") 
                        
                        #is_complement = True if "VP" in align_pos_list  else False
                                   
                        
                        
                #print("cur_node: ", cur_node.leaf_labels())
                #print("remove_token", removed_token)
                
                # pp_idx = children_pos_list.index("PP")
                # np_idx = children_pos_list.index("NP")
                # if key_entity in cur_node.children[np_idx].leaf_labels():
                #     # entity가 NP(명사절)에 들어있음. 명사에 대한 수식
                #     entity_found = True
                #     removed_tokens = cur_node.children[pp_idx].leaf_labels()
                #     cur_node.children.remove(cur_node.children[pp_idx])
                # elif key_entity in cur_node.children[pp_idx].leaf_labels():
                #     # entity가 PP(수식절) 안에 들어있을 경우. 수식에 대한 수식
                #     pp_node = cur_node.parent
                #     pp_parent_child_pos_list = [child.pos for child in pp_node.parent.children]
                #     pp_node_idx = pp_node.parent.children.index(pp_node)
                #     if pp_parent_child_pos_list.count("PP") > 1 and \
                #         pp_node_idx + 1 < len(pp_parent_child_pos_list) and \
                #         pp_parent_child_pos_list[pp_node_idx + 1] == "PP":
                #         # entity를 포함한 수식 절을 수식하는 수식절이 있어야 수식에 대한 수식 성립
                #         pp_idx = pp_node_idx + 1 # 수식절이 수식절의 뒤에 있다고 가정
                #         entity_found = True
                #         removed_tokens = pp_node.parent.children[pp_idx].leaf_labels()
                #         pp_node.parent.children.remove(pp_node.parent.children[pp_idx])
                                    
                        
        # print("removed_tokens:", removed_tokens)               
        # print("-------------------------------------------")
        return removed_tokens

def process_ambiguous_questions_set(file_path, output_path):
    tree = ConstituencyTree()
    ds_json = {}
    
    with open(file_path, mode="r+") as fp:
        ds_json = json.loads(fp.read())
    
    
    pro_bar = tqdm.tqdm(
        desc="rebuilding",
        total=len(ds_json),
        position=0,
        leave=True   
    )
    changed_json = {}
    changed_count = 0
    for idx, k in enumerate(ds_json.keys()):
        q = ds_json[k]["question"]
        tree.sentence_to_tree(q)
        removed_tokens = []
        
        entities = [word for _, word in ds_json[k]["irrelated_object_names"].items()]
        entities = list(set(entities))
        
        # print(q)
        for e in entities:
            rem_result = tree.remove_modifier(e)
            if rem_result:
                removed_tokens.append(rem_result)
                tree.sentence_tree.renew_labels()
        
        if len(removed_tokens) > 0:
            changed_count += 1
        leaf_labels = tree.sentence_tree.renew_labels()
        changed_sentence = ' '.join(leaf_labels[:-1])
        changed_sentence += leaf_labels[-1] # ? 문장부호 붙이도록 처리
        changed_json[k] = {
            "question": changed_sentence,
            "original_question": q,
            "removed": removed_tokens, # [' '.join(tokens) for tokens in removed_tokens],
            "irrelated_objects": entities
        }
        
        #     print(changed_json)
            # exit()
        
        pro_bar.update()
        
        if idx > 299:
          break
    
    
    print("# of changed sentence: {}".format(changed_count))
    with open(output_path, mode="wt", encoding="utf-8") as fp:
        fp.write(json.dumps(changed_json))
        
if __name__ == "__main__":
    # "What is the man (in the street) wearing?" # 뒤에 오는 구로 수식
    # "Is the (silver) car to the right of a man?" # 형용사 수식
    # "Is the (yellow) helmet to the right or to the left of the man (on the right)?"
    # "What is the item of furniture to the left of the drawer (that is on the bottom of the gas stove)?" that절 수식
    # "Is the chair to the right or to the left of the pillow (that looks brown)?"
    # "Is the (red) chair on top of the elephant (near the tree)?" # 형용사, 구 수식, 구 불가
    # "What is the woman that is to the right of the bucket wearing?" wsj_bert에서 잘못된 트리 구조
    print("main")
    '''sample_setence = "Are the ripe bananas above a newspaper?"
    
    tree = ConstituencyTree()
    
    tree.sentence_to_tree(sample_setence)
    
    tree.remove_modifier("bananas")
    #tree.remove_modifier("elephant")
    
    print(tree.sentence_tree.renew_labels())'''
    
    process_ambiguous_questions_set("./ambiguous_questions.json", "./ambiguous_questions_rebuilt_300.json")