#!/usr/bin/env python3

import sys
import os
Set = set

DFILE_KEYWORDS = ["requirements", "types", "constants", "predicates", "functions", "action", "private"]
DFILE_REQ_KEYWORDS = ["typing","strips","multi-agent","unfactored-privacy","fluents","numeric-fluents"]
DFILE_SUBKEYWORDS = ["parameters", "precondition", "effect", "duration"]
PFILE_KEYWORDS = ["objects", "init", "goal","private", "metric"]
AFILE_KEYWORDS = ["agents"]

verbose = False


class Predicate(object):
  """A loose interpretation of a predicate used for all similar collections.

  Without a name it is a parameter list.
  It can be typed (or not).
    If typed then args = [[var, type], ...]
    Else args = [var, ...]
  It can be negated.
  It may contain variables or objects in its arguments.
  """
  def __init__(self, name, args, is_typed, is_negated):
    self.name = name
    self.args = args
    self.arity = len(args)
    self.is_typed = is_typed
    self.is_negated = is_negated
    self.ground_facts = set()
    self.agent_param = -1

  def pddl_rep(self):
    """Returns the PDDL version of the instance."""
    rep = ''
    if self.is_negated:
      rep += "(not "
    if self.name != "":
      rep += "(" + self.name + " "
    else:
      rep += "("
    for argument in self.args:
      if self.is_typed:
        rep += argument[0] + " - " + argument[1] + " "
      else:
        rep += argument + " "
    rep = rep[:-1]
    rep += ")"
    if self.is_negated:
      rep += ")"
    return rep

  def __repr__(self):
    return self.pddl_rep()



class Action(object):
  """Represents a simple non-temporal action."""
  def __init__(self, name, parameters, precondition, effect):
    self.name = name
    self.parameters = parameters
    self.precondition = precondition
    self.effect = effect
    self.duration = 1
    self.agent = ""
    self.agent_type = ""

  def pddl_rep(self):
    """Returns the PDDL version of the instance."""
    rep = ''
    rep += "(:action " + self.name + "\n"
    rep += "\t:parameters " + str(self.parameters) + "\n"
    if len(self.precondition) > 1:
      rep += "\t:precondition (and\n"
    else:
      rep += "\t:precondition \n"
    for precon in self.precondition:
      rep += "\t\t" + str(precon) + "\n"
    if len(self.precondition) > 1:
      rep += "\t)\n"
    if len(self.effect) > 1:
      rep += "\t:effect (and\n"
    else:
      rep += "\t:effect \n"
    for eff in self.effect:
      rep += "\t\t" + str(eff) + "\n"
    if len(self.effect) > 1:
      rep += "\t)\n"
    rep += ")\n"
    return rep

  def __repr__(self):
    return self.name #+ str(self.parameters)



class PlanningProblem(object):
  def __init__(self, domainfile, problemfile):
    self.domain = '' #String
    self.requirements = set() #[String]
    self.type_list = Set() #{String}
    self.type_list.add('object')
    self.types = {} #Key = supertype_name, Value = type
    self.predicates = [] #[Predicate]
    self.constants = [] #[Constant definitions]
    self.functions = [] #[Function definitions]
    self.actions = [] #[Action]
    self.agent_types = set()
    self.agents = set()
    self.problem = '' #String
    self.object_list = Set() #{String}
    self.objects = {} #Key = type, Value = object_name
    self.init = [] #List of Predicates
    self.init_functions = [] #List of function initial values
    self.goal = [] #List of Predicates
    
    self.parse_domain(domainfile)
    self.parse_problem(problemfile)

    for t in self.agent_types:
      if t in self.objects:
        self.agents = self.agents | set(self.objects[t])
    
    self.requirements = self.requirements - {"multi-agent","unfactored-privacy"}
    
  def parse_domain(self, domainfile):
    """Parses a PDDL domain file."""
    
    with open(domainfile) as dfile:
      dfile_array = self._get_file_as_array(dfile)
    #Deal with front/end define, problem, :domain
    if dfile_array[0:4] != ['(', 'define', '(', 'domain']:
      print('PARSING ERROR: Expected (define (domain ... at start of domain file')
      sys.exit()
    self.domain = dfile_array[4]

    dfile_array = dfile_array[6:-1]
    
    # First pass: extract raw functions section before tokenizing
    self._extract_raw_functions(domainfile)
    
    opencounter = 0
    keyword = ''
    obj_list = []
    is_obj_list = True
    for word in dfile_array:
      if word == '(':
        opencounter += 1
      elif word == ')':
        opencounter -= 1
      elif word.startswith(':'):
        if word[1:] not in DFILE_KEYWORDS:
          pass
        elif keyword != 'requirements':
          keyword = word[1:]
      if opencounter == 0:
        if keyword == 'action':
          self.actions.append(obj_list)
          obj_list = []
        if keyword == 'types':
          for element in obj_list:
            self.types.setdefault('object', []).append(element)
            self.type_list.add('object')
            self.type_list.add(element)
          obj_list = []
        if keyword == 'constants':
          if len(obj_list) > 0:
            const_str = ''
            for element in obj_list:
              const_str += element + ' '
            const_str = const_str.rstrip()
            if const_str:
              self.constants.append(const_str)
            obj_list = []
        keyword = ''

      if keyword == 'requirements': #Requirements list
        if word != ':requirements':
          if not word.startswith(':'):
            print('PARSING ERROR: Expected requirement to start with :')
            sys.exit()
          elif word[1:] not in DFILE_REQ_KEYWORDS:
            print('WARNING: Unknown Rquierement ' + word[1:])
            #print 'Requirements must only be: ' + str(DFILE_REQ_KEYWORDS)
            #sys.exit()
          else:
            self.requirements.add(word[1:])
      elif keyword == 'action':
        obj_list.append(word)
      elif not word.startswith(':'):
        if keyword == 'types': #Typed list of objects
          if is_obj_list:
            if word == '-':
              is_obj_list = False
            else:
              obj_list.append(word)
          else:
            #word is type
            for element in obj_list:
              if not word in self.type_list:
                self.types.setdefault('object', []).append(word)
                self.type_list.add(word)
              self.types.setdefault(word, []).append(element)
              self.type_list.add(element)
              self.type_list.add(word)
            is_obj_list = True
            obj_list = []
        elif keyword == 'constants': #Typed list of constants (like types but for constants)
          if is_obj_list:
            if word == '-':
              is_obj_list = False
            else:
              obj_list.append(word)
          else:
            #word is type
            const_str = ''
            for element in obj_list:
              const_str += element + ' '
            const_str = const_str.rstrip() + ' - ' + word
            self.constants.append(const_str)
            is_obj_list = True
            obj_list = []
        elif keyword == 'predicates' or keyword == 'private': #Internally typed predicates
          if word == ')':
            if keyword == 'private':
              #print "...skip agent: " +  str(obj_list[:3])
              obj_list = obj_list[3:]
              keyword = 'predicates'
            if len(obj_list) == 0:
              #print "...skip )"
              continue
            p_name = obj_list[0]
            #print "parse predicate: " + p_name + " " + str(obj_list)
            pred_list = self._parse_name_type_pairs(obj_list[1:],self.type_list)
            self.predicates.append(Predicate(p_name, pred_list, True, False))
            obj_list = []
          elif word != '(':
            obj_list.append(word)

    #Work on the actions
    new_actions = []
    for action in self.actions:
      act_name = action[1]
      act = {}
      action = action[2:]
      keyword = ''
      for word in action:
        if word.startswith(':'):
          keyword = word[1:]
        else:
          act.setdefault(keyword, []).append(word)
      self.agent_types.add(act.get('agent')[2])
      agent = self._parse_name_type_pairs(act.get('agent'),self.type_list)
      param_list = agent + self._parse_name_type_pairs(act.get('parameters')[1:-1],self.type_list)
      up_params = Predicate('', param_list, True, False)
      pre_list = self._parse_unground_propositions(act.get('precondition'))
      eff_list = self._parse_unground_propositions(act.get('effect'))
      new_act = Action(act_name, up_params, pre_list, eff_list)

      new_actions.append(new_act)
    self.actions = new_actions

  def parse_problem(self, problemfile):
    """The main method for parsing a PDDL files."""

    with open(problemfile) as pfile:
      pfile_array = self._get_file_as_array(pfile)
    #Deal with front/end define, problem, :domain
    if pfile_array[0:4] != ['(', 'define', '(', 'problem']:
      print('PARSING ERROR: Expected (define (problem ... at start of problem file')
      sys.exit()
    self.problem = pfile_array[4]
    if pfile_array[5:8] != [')', '(', ':domain']:
      print('PARSING ERROR: Expected (:domain ...) after (define (problem ...)')
      sys.exit()
    if self.domain != pfile_array[8]:
      print('ERROR - names don\'t match between domain and problem file.')
      #sys.exit()
    if pfile_array[9] != ')':
      print('PARSING ERROR: Expected end of domain declaration')
      sys.exit()
    pfile_array = pfile_array[10:-1]

    opencounter = 0
    keyword = ''
    is_obj_list = True
    obj_list = []
    int_obj_list = []
    int_opencounter = 0
    init_depth = 0  # Track depth for init parsing
    for word in pfile_array:
      if word == '(':
        opencounter += 1
        if keyword == 'init':
          init_depth += 1
      elif word == ')':
        if keyword == 'objects':
          obj_list = []
        opencounter -= 1
        if keyword == 'init':
          init_depth -= 1
      elif word.startswith(':'):
        if word[1:] not in PFILE_KEYWORDS:
          print('PARSING ERROR: Unknown keyword: ') + word[1:]
          print('Known keywords: ') + str(PFILE_KEYWORDS)
        else:
          keyword = word[1:]
      if opencounter == 0:
        keyword = ''

      if not word.startswith(':'):
        if keyword == 'objects' or keyword == 'private': #Typed list of objects
          #print "word: " + word
          #print "obj_list: " + str(obj_list)
          if keyword == 'private':
              #print "...skip agent: " +  word
              obj_list = []
              keyword = 'objects'
              continue
          if is_obj_list:
            if word == '-':
              is_obj_list = False
            else:
              obj_list.append(word)
          else:
            #word is type
            for element in obj_list:
              if word in self.type_list:
                self.objects.setdefault(word, []).append(element)
                self.object_list.add(element)
              else:
                print(self.type_list)
                print("ERROR unknown type " + word)
                sys.exit()
            is_obj_list = True
            obj_list = []
        elif keyword == 'init':
          # Collect all tokens for this init fact/function
          if word == ')' and init_depth == 0:
              if len(obj_list) > 0:
                # Check if this is a numeric function initialization (= ...)
                if obj_list[0] == '=':
                  # obj_list is ['=', 'travel-slow', 'n0', 'n1', '6'] 
                  # Assuming no parens were collected (we skip them below)
                  # We need to reconstruct with inner parens for function call
                  # Format: = (func arg1 arg2 ...) value
                  # We don't know exactly where function ends and value starts
                  # PDDL numeric fact format: (= (function-call) value)
                  # For elevators: (= (travel-slow n0 n1) 6)
                  # obj_list would be ['=', 'travel-slow', 'n0', 'n1', '6']
                  # So last item is value, everything before is function + args
                  init_str = '= ( '
                  for i in range(1, len(obj_list)-1):
                    init_str += obj_list[i] + ' '
                  init_str = init_str.rstrip() + ' ) ' + obj_list[-1]
                  self.init_functions.append(init_str)
                else:
                  # Regular predicate - just name and args, no parens
                  self.init.append(Predicate(obj_list[0], obj_list[1:],
                                   False, False))
              obj_list = []
          elif word != '(' and word != ')':
            # Only collect non-paren tokens (parens are just structure)
            obj_list.append(word)
        elif keyword == 'goal':
          if word == '(':
            int_opencounter += 1
          elif word == ')':
            int_opencounter -= 1
          obj_list.append(word)
          if int_opencounter == 0:
              self.goal = self._parse_unground_propositions(obj_list)
              obj_list = []

  def get_type_of_object(self,obj):
    for t in self.objects.keys():
      if obj in self.objects[t]:
        return t

  def print_domain(self):
    """Prints out the planning problem in (semi-)readable format."""
    print('\n*****************')
    print('DOMAIN: ') + self.domain
    print('REQUIREMENTS: ') + str(self.requirements)
    print('TYPES: ') + str(self.types)
    print('PREDICATES: ') + str(self.predicates)
    print('ACTIONS: ') + str(self.actions)
    print('****************')

  def print_problem(self):
    """Prints out the planning problem in (semi-)readable format."""
    print('\n*****************')
    print('PROBLEM: ') + self.problem
    print('OBJECTS: ') + str(self.objects)
    print('INIT: ') + str(self.init)
    print('GOAL: ') + str(self.goal)
    print('AGENTS: ') + str(self.agents)
    print('****************')

  def _format_function_def(self, tokens):
    """Format a function definition from parsed tokens.
    
    Handles both simple functions and typed parameter functions.
    Example: ['total-cost', '-', 'number'] -> 'total-cost - number'
    Example: ['travel-slow', '?f1', '-', 'count', '?f2', '-', 'count', '-', 'number']
             -> 'travel-slow ?f1 - count ?f2 - count - number'
    """
    if len(tokens) == 0:
      return None
    
    result = []
    i = 0
    while i < len(tokens):
      # Collect parameter with its type if follows pattern: param - type
      if i + 2 < len(tokens) and tokens[i + 1] == '-':
        result.append(tokens[i])
        result.append('-')
        result.append(tokens[i + 2])
        i += 3
      else:
        result.append(tokens[i])
        i += 1
    
    return ' '.join(result)

  def _extract_raw_functions(self, domainfile):
    """Extract functions section from domain file as raw text."""
    try:
      with open(domainfile, 'r') as f:
        content = f.read()
      
      # Find :functions section
      func_start = content.find(':functions')
      if func_start == -1:
        return
      
      # Find the opening paren before :functions
      paren_pos = content.rfind('(', 0, func_start)
      
      # Find matching closing paren
      paren_count = 1
      pos = paren_pos + 1
      while pos < len(content) and paren_count > 0:
        if content[pos] == '(':
          paren_count += 1
        elif content[pos] == ')':
          paren_count -= 1
        pos += 1
      
      # Extract and parse the functions section
      func_section = content[paren_pos:pos]
      # Remove :functions keyword and outer parens
      func_content = func_section.replace(':functions', '').strip()
      if func_content.startswith('('):
        func_content = func_content[1:]
      if func_content.endswith(')'):
        func_content = func_content[:-1]
      
      # Parse individual function definitions
      # Each function def is like: (total-cost) - number
      lines = func_content.split('\n')
      current_func = ''
      for line in lines:
        line = line.strip()
        if not line:
          continue
        current_func += line + ' '
        # Check if we have a complete function def (ends with type after -)
        if ' - ' in current_func and current_func.count('(') == current_func.count(')'):
          self.functions.append(current_func.strip())
          current_func = ''
    except:
      pass  # If extraction fails, continue without functions
  
 

         
    
  #Get string of file with comments removed - comments are rest of line after ';'
  def _get_file_as_array(self, file_):
    """Returns the file split into array of words.

    Removes comments and separates parenthesis.
    """
    file_as_string = ""
    for line in file_:
      if ";" in line:
        line = line[:line.find(";")]
      line = (line.replace('\t', '').replace('\n', ' ')
          .replace('(', ' ( ').replace(')', ' ) '))
      file_as_string += line
    file_.close()
    return file_as_string.strip().split()

  def _parse_name_type_pairs(self, array, types):
    """Parses array creating paris of form (name, type).

    Expects array such as [?a, -, agent, ...]."""
    pred_list = []
    if len(array)%3 != 0:
      print("Expected predicate to be typed " + str(array))
      sys.exit()
    for i in range(0, len(array)//3):
      if array[3*i+1] != '-':
        print("Expected predicate to be typed")
        sys.exit()
      if array[3*i+2] in types:
        pred_list.append((array[3*i], array[3*i+2]))
      else:
        print("PARSING ERROR {} not in types list".format(array[3*i+2]))
        print("Types list: {}".format(self.type_list))
        sys.exit()
    return pred_list

  def _parse_unground_proposition(self, array):
    """Parses a variable proposition returning dict."""
    negative = False
    if array[1] == 'not':
      negative = True
      array = array[2:-1]
    return Predicate(array[1], array[2:-1], False, negative)

  def _parse_unground_propositions(self, array):
    """Parses possibly conjunctive list of unground propositions.

    Expects array such as [(and, (, at, ?a, ?x, ), ...].
    """
    prop_list = []
    if array[0:3] == ['(', 'and', '(']:
      array = array[2:-1]
    #Split array into blocks
    opencounter = 0
    prop = []
    for word in array:
      if word == '(':
        opencounter += 1
      if word == ')':
        opencounter -= 1
      prop.append(word)
      if opencounter == 0:
        prop_list.append(self._parse_unground_proposition(prop))
        prop = []
    #print array[:array.index(')') + 1]
    return prop_list

  def write_pddl_domain(self, output_file):
    """Writes an unfactored MA-PDDL domain file for this planning problem."""
    file_ = open(output_file, 'w')
    to_write = "(define (domain " + self.domain + ")\n"
    #Requirements
    to_write += "\t(:requirements"
    for r in self.requirements:
      to_write += " :"+r
    to_write += ")\n"
    #Types
    to_write += "(:types\n"
    for type_ in self.types:
      to_write += "\t"
      for key in self.types.get(type_):
        to_write += key + " "
      to_write += "- " + type_
      to_write += "\n"
    to_write += ")\n"
    #Constants
    if len(self.constants) > 0:
      to_write += "(:constants\n"
      for const_str in self.constants:
        to_write += "\t" + const_str + "\n"
      to_write += ")\n"
    #Public predicates
    to_write += "(:predicates\n"
    for predicate in self.predicates:
      to_write += "\t{}\n".format(predicate.pddl_rep())
    to_write += ")\n"
    #Functions
    if len(self.functions) > 0:
      to_write += "(:functions\n"
      for func in self.functions:
        to_write += "\t{}\n".format(func)
      to_write += ")\n"
    #Actions
    for action in self.actions:
      to_write += "\n{}\n".format(action.pddl_rep())
    
    #Endmatter
    to_write += ")" #Close domain defn
    file_.write(to_write)
    file_.close()

  def write_pddl_problem(self, output_file):
    file_ = open(output_file, 'w')
    to_write = "(define (problem " + self.problem +") "
    to_write += "(:domain " + self.domain + ")\n"
    #Objects
    to_write += "(:objects\n"
    for obj in self.object_list:
      to_write += "\t" + obj + " - " + self.get_type_of_object(obj) + "\n"
    to_write += ")\n"
    to_write += "(:init\n"
    for predicate in self.init:
      to_write += "\t{}\n".format(predicate)
    for func_init in self.init_functions:
      to_write += "\t({})\n".format(func_init)
    to_write += ")\n"
    to_write += "(:goal\n\t(and\n"
    for goal in self.goal:
      to_write += "\t\t{}\n".format(goal)
    to_write += "\t)\n)\n"
    #to_write += "(:metric minimize (total-time))\n" #TODO
    #Endmatter
    to_write += ")"
    file_.write(to_write)
    file_.close()

  def write_addl(self, output_file):
    file_ = open(output_file, 'w')
    to_write = "(define (problem " + self.problem +") "
    to_write += "(:domain " + self.domain + ")\n"
    #Objects
    to_write += "(:agents"
    for obj in self.agents:
      to_write += " " + obj 
    to_write += ")\n"
    to_write += ")"
    file_.write(to_write)
    file_.close()

  def write_agent_list(self, output_file):
    file_ = open(output_file, 'w')
    to_write = ""
    for obj in self.agents:
      to_write += obj + "\n"
    file_.write(to_write)
    file_.close()

  


if __name__ == "__main__":
  if len(sys.argv) < 4:
    print('Requires 2 args')
    print('arg1: folder')
    print('arg2: domain')
    print('arg3: problem')
    print('arg4: output folder')
  else:
    pp = PlanningProblem(sys.argv[1] + "/" + sys.argv[2] + ".pddl", sys.argv[1] + "/" + sys.argv[3] + ".pddl")
    
    if verbose:
      pp.print_domain()
      pp.print_problem()

    if not os.path.exists(sys.argv[4]):
      os.mkdir(sys.argv[4])

    pp.write_pddl_domain(sys.argv[4] + "/" + sys.argv[2] + ".pddl")
    pp.write_pddl_problem(sys.argv[4] + "/" + sys.argv[3] + ".pddl")
    pp.write_addl(sys.argv[4] + "/" + sys.argv[3] + ".addl")
    pp.write_agent_list(sys.argv[4] + "/" + sys.argv[3] + ".agents")

    




