import ast
ImportFrom = ast.ImportFrom
alias = ast.alias
ClassDef = ast.ClassDef
FunctionDef = ast.FunctionDef
arg = ast.arg
arguments = ast.arguments
Import = ast.Import
Assign = ast.Assign
Constant = ast.Constant
Return = ast.Return
Call = ast.Call
keyword = ast.keyword
Name = ast.Name
Load = ast.Load
Store = ast.Store
Dict = ast.Dict
Expr = ast.Expr
BinOp = ast.BinOp
Attribute = ast.Attribute
Mult= ast.Mult
Subscript = ast.Subscript
UnaryOp = ast.UnaryOp
USub = ast.USub
JoinedStr=ast.JoinedStr
FormattedValue=ast.FormattedValue
Compare = ast.Compare
If = ast.If 
List = ast.List
For = ast.For
Eq = ast.Eq
AnnAssign = ast.AnnAssign
Tuple = ast.Tuple
In = ast.In
IfExp = ast.IfExp
Try = ast.Try
ExceptHandler = ast.ExceptHandler
Pass = ast.Pass

x = ast.Module(
  body=[
    ImportFrom(
      module='typing',
      names=[
        alias(name='Dict'),
        alias(name='Any'),
        alias(name='List'),
        alias(name='Optional'),
        alias(name='Literal')],
      level=0),
    ImportFrom(
      module='langgraph.graph',
      names=[
        alias(name='StateGraph'),
        alias(name='START'),
        alias(name='END'),
        alias(name='MessagesState')],
      level=0),
    ImportFrom(
      module='langgraph.checkpoint.memory',
      names=[
        alias(name='InMemorySaver')],
      level=0),
    ImportFrom(
      module='langchain_openai',
      names=[
        alias(name='ChatOpenAI')],
      level=0),
    ImportFrom(
      module='pydantic',
      names=[
        alias(name='BaseModel'),
        alias(name='Field')],
      level=0),
    ImportFrom(
      module='langchain_core.tools',
      names=[
        alias(name='tool')],
      level=0),
    ImportFrom(
      module='langchain_core.messages',
      names=[
        alias(name='HumanMessage'),
        alias(name='AIMessage'),
        alias(name='SystemMessage')],
      level=0),
    Import(
      names=[
        alias(name='re')]),
    Import(
      names=[
        alias(name='json')]),
    ImportFrom(
      module='langchain.agents',
      names=[
        alias(name='create_react_agent')],
      level=0),
    FunctionDef(
      name='calorie_lookup_tool',
      args=arguments(
        args=[
          arg(
            arg='food_item',
            annotation=Name(id='str', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to look up calorie information for various food items. Input: food item (str). Output: estimated calories (int).')),
        Assign(
          targets=[
            Name(id='food_calories', ctx=Store())],
          value=Dict(
            keys=[
              Constant(value='banana'),
              Constant(value='apple'),
              Constant(value='chicken breast'),
              Constant(value='rice')],
            values=[
              Constant(value=105),
              Constant(value=95),
              Constant(value=165),
              Constant(value=130)])),
        Return(
          value=Call(
            func=Attribute(
              value=Name(id='food_calories', ctx=Load()),
              attr='get',
              ctx=Load()),
            args=[
              Call(
                func=Attribute(
                  value=Name(id='food_item', ctx=Load()),
                  attr='lower',
                  ctx=Load())),
              Constant(value=0)]))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Name(id='int', ctx=Load())),
    FunctionDef(
      name='exercise_calorie_estimator_tool',
      args=arguments(
        args=[
          arg(
            arg='exercise_type',
            annotation=Name(id='str', ctx=Load())),
          arg(
            arg='duration_minutes',
            annotation=Name(id='int', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to estimate calories burned for various exercise types and durations. Input: exercise type (str), duration in minutes (int). Output: estimated calories burned (int).')),
        Assign(
          targets=[
            Name(id='exercise_calories_per_minute', ctx=Store())],
          value=Dict(
            keys=[
              Constant(value='running'),
              Constant(value='walking'),
              Constant(value='swimming'),
              Constant(value='cycling')],
            values=[
              Constant(value=10),
              Constant(value=5),
              Constant(value=7),
              Constant(value=8)])),
        Return(
          value=BinOp(
            left=Call(
              func=Attribute(
                value=Name(id='exercise_calories_per_minute', ctx=Load()),
                attr='get',
                ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='exercise_type', ctx=Load()),
                    attr='lower',
                    ctx=Load())),
                Constant(value=0)]),
            op=Mult(),
            right=Name(id='duration_minutes', ctx=Load())))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Name(id='int', ctx=Load())),
    FunctionDef(
      name='store_food_intake_tool',
      args=arguments(
        args=[
          arg(
            arg='date',
            annotation=Name(id='str', ctx=Load())),
          arg(
            arg='food_item',
            annotation=Name(id='str', ctx=Load())),
          arg(
            arg='calories',
            annotation=Name(id='int', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to store calorie intake data (food items and calories) into a database. Input: date (str), food item (str), calories (int). Output: success/failure status (str).')),
        Expr(
          value=Call(
            func=Name(id='print', ctx=Load()),
            args=[
              JoinedStr(
                values=[
                  Constant(value='Storing food intake: Date: '),
                  FormattedValue(
                    value=Name(id='date', ctx=Load()),
                    conversion=-1),
                  Constant(value=', Item: '),
                  FormattedValue(
                    value=Name(id='food_item', ctx=Load()),
                    conversion=-1),
                  Constant(value=', Calories: '),
                  FormattedValue(
                    value=Name(id='calories', ctx=Load()),
                    conversion=-1)])])),
        Return(
          value=Constant(value='success'))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Name(id='str', ctx=Load())),
    FunctionDef(
      name='store_exercise_data_tool',
      args=arguments(
        args=[
          arg(
            arg='date',
            annotation=Name(id='str', ctx=Load())),
          arg(
            arg='exercise_type',
            annotation=Name(id='str', ctx=Load())),
          arg(
            arg='duration_minutes',
            annotation=Name(id='int', ctx=Load())),
          arg(
            arg='calories_burned',
            annotation=Name(id='int', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to store exercise data (type, duration, calories burned) into a database. Input: date (str), exercise type (str), duration minutes (int), calories burned (int). Output: success/failure status (str).')),
        Expr(
          value=Call(
            func=Name(id='print', ctx=Load()),
            args=[
              JoinedStr(
                values=[
                  Constant(value='Storing exercise data: Date: '),
                  FormattedValue(
                    value=Name(id='date', ctx=Load()),
                    conversion=-1),
                  Constant(value=', Type: '),
                  FormattedValue(
                    value=Name(id='exercise_type', ctx=Load()),
                    conversion=-1),
                  Constant(value=', Duration: '),
                  FormattedValue(
                    value=Name(id='duration_minutes', ctx=Load()),
                    conversion=-1),
                  Constant(value=' mins, Calories Burned: '),
                  FormattedValue(
                    value=Name(id='calories_burned', ctx=Load()),
                    conversion=-1)])])),
        Return(
          value=Constant(value='success'))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Name(id='str', ctx=Load())),
    FunctionDef(
      name='get_food_history_tool',
      args=arguments(
        args=[
          arg(
            arg='query_date_range',
            annotation=Name(id='str', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to retrieve historical food consumption data from the database based on a date or date range. Input: query_date_range (str). Output: list of food entries (list[dict]).')),
        If(
          test=Compare(
            left=Name(id='query_date_range', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='yesterday')]),
          body=[
            Return(
              value=List(
                elts=[
                  Dict(
                    keys=[
                      Constant(value='date'),
                      Constant(value='item'),
                      Constant(value='calories')],
                    values=[
                      Constant(value='2023-10-26'),
                      Constant(value='banana'),
                      Constant(value=105)])],
                ctx=Load()))],
          orelse=[
            If(
              test=Compare(
                left=Name(id='query_date_range', ctx=Load()),
                ops=[
                  Eq()],
                comparators=[
                  Constant(value='Monday')]),
              body=[
                Return(
                  value=List(
                    elts=[
                      Dict(
                        keys=[
                          Constant(value='date'),
                          Constant(value='item'),
                          Constant(value='calories')],
                        values=[
                          Constant(value='2023-10-23'),
                          Constant(value='sandwich'),
                          Constant(value=400)]),
                      Dict(
                        keys=[
                          Constant(value='date'),
                          Constant(value='item'),
                          Constant(value='calories')],
                        values=[
                          Constant(value='2023-10-23'),
                          Constant(value='apple'),
                          Constant(value=95)])],
                    ctx=Load()))])]),
        Return(
          value=List(ctx=Load()))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Subscript(
        value=Name(id='list', ctx=Load()),
        slice=Name(id='dict', ctx=Load()),
        ctx=Load())),
    FunctionDef(
      name='get_exercise_history_tool',
      args=arguments(
        args=[
          arg(
            arg='query_date_range',
            annotation=Name(id='str', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='A tool to retrieve historical exercise data from the database based on a date or date range. Input: query_date_range (str). Output: list of exercise entries (list[dict]).')),
        If(
          test=Compare(
            left=Name(id='query_date_range', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='yesterday')]),
          body=[
            Return(
              value=List(
                elts=[
                  Dict(
                    keys=[
                      Constant(value='date'),
                      Constant(value='type'),
                      Constant(value='duration_minutes'),
                      Constant(value='calories_burned')],
                    values=[
                      Constant(value='2023-10-26'),
                      Constant(value='running'),
                      Constant(value=30),
                      Constant(value=350)])],
                ctx=Load()))],
          orelse=[
            If(
              test=Compare(
                left=Name(id='query_date_range', ctx=Load()),
                ops=[
                  Eq()],
                comparators=[
                  Constant(value='last week')]),
              body=[
                Return(
                  value=List(
                    elts=[
                      Dict(
                        keys=[
                          Constant(value='date'),
                          Constant(value='type'),
                          Constant(value='duration_minutes'),
                          Constant(value='calories_burned')],
                        values=[
                          Constant(value='2023-10-23'),
                          Constant(value='running'),
                          Constant(value=20),
                          Constant(value=250)]),
                      Dict(
                        keys=[
                          Constant(value='date'),
                          Constant(value='type'),
                          Constant(value='duration_minutes'),
                          Constant(value='calories_burned')],
                        values=[
                          Constant(value='2023-10-25'),
                          Constant(value='walking'),
                          Constant(value=60),
                          Constant(value=300)])],
                    ctx=Load()))])]),
        Return(
          value=List(ctx=Load()))],
      decorator_list=[
        Name(id='tool', ctx=Load())],
      returns=Subscript(
        value=Name(id='list', ctx=Load()),
        slice=Name(id='dict', ctx=Load()),
        ctx=Load())),
    ClassDef(
      name='GraphState',
      bases=[
        Name(id='MessagesState', ctx=Load())],
      body=[
        Expr(
          value=Constant(value=' The GraphState represents the state of the LangGraph workflow.\n    Below is the definition of MessagesState, the AnyMessage refers to AIMessage, HumanMessage, or SystemMessage etc.\n    the add_messages is a reducer, which means that when doing return {{"messages": [AIMessage(content="...")]}}, it will append the new message to the messages variable and not override it..\n    class MessagesState(TypedDict):\n        messages: Annotated[list[AnyMessage], add_messages]\n    ')),
        AnnAssign(
          target=Name(id='user_input', ctx=Store()),
          annotation=Name(id='str', ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='description',
                value=Constant(value="The user's original query."))]),
          simple=1),
        AnnAssign(
          target=Name(id='intent', ctx=Store()),
          annotation=Name(id='str', ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='description',
                value=Constant(value="The identified intent of the user's query (e.g., 'track_food', 'track_exercise', 'get_diet_history', 'get_exercise_history', 'get_net_effect')."))]),
          simple=1),
        AnnAssign(
          target=Name(id='food_items', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Subscript(
              value=Name(id='list', ctx=Load()),
              slice=Name(id='dict', ctx=Load()),
              ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="A list of dictionaries, each containing 'item' (str) and 'quantity' (str)."))]),
          simple=1),
        AnnAssign(
          target=Name(id='calculated_calories', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='int', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value='The total calculated calories for the food intake.'))]),
          simple=1),
        AnnAssign(
          target=Name(id='storage_status', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='str', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="Status of data storage (e.g., 'success', 'failure')."))]),
          simple=1),
        AnnAssign(
          target=Name(id='exercise_details', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='dict', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="A dictionary containing 'type' (str) and 'duration_minutes' (int)."))]),
          simple=1),
        AnnAssign(
          target=Name(id='estimated_calories_burned', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='int', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value='The estimated calories burned during the exercise.'))]),
          simple=1),
        AnnAssign(
          target=Name(id='query_date_range', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='str', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="The date or date range for the historical query (e.g., 'yesterday', 'last week', 'Monday')."))]),
          simple=1),
        AnnAssign(
          target=Name(id='diet_history_results', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Subscript(
              value=Name(id='list', ctx=Load()),
              slice=Name(id='dict', ctx=Load()),
              ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="A list of dictionaries, each representing a food entry with 'date', 'item', and 'calories'."))]),
          simple=1),
        AnnAssign(
          target=Name(id='exercise_history_results', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Subscript(
              value=Name(id='list', ctx=Load()),
              slice=Name(id='dict', ctx=Load()),
              ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="A list of dictionaries, each representing an exercise entry with 'date', 'type', 'duration_minutes', and 'calories_burned'."))]),
          simple=1),
        AnnAssign(
          target=Name(id='query_period', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='str', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None)),
              keyword(
                arg='description',
                value=Constant(value="The period for the net calorie calculation (e.g., 'today', 'this week', 'last month')."))]),
          simple=1),
        AnnAssign(
          target=Name(id='total_calories_consumed', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='int', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None))]),
          simple=1),
        AnnAssign(
          target=Name(id='total_calories_burned', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='int', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None))]),
          simple=1),
        AnnAssign(
          target=Name(id='net_calorie_balance', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Optional', ctx=Load()),
            slice=Name(id='int', ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='default',
                value=Constant(value=None))]),
          simple=1)]),
    ClassDef(
      name='IntentClassification',
      bases=[
        Name(id='BaseModel', ctx=Load())],
      body=[
        Expr(
          value=Constant(value='Structured output for intent classification.')),
        AnnAssign(
          target=Name(id='intent', ctx=Store()),
          annotation=Subscript(
            value=Name(id='Literal', ctx=Load()),
            slice=Tuple(
              elts=[
                Constant(value='track_food'),
                Constant(value='track_exercise'),
                Constant(value='get_diet_history'),
                Constant(value='get_exercise_history'),
                Constant(value='get_net_effect')],
              ctx=Load()),
            ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='description',
                value=Constant(value='Classified intent'))]),
          simple=1),
        AnnAssign(
          target=Name(id='reasoning', ctx=Store()),
          annotation=Name(id='str', ctx=Load()),
          value=Call(
            func=Name(id='Field', ctx=Load()),
            keywords=[
              keyword(
                arg='description',
                value=Constant(value='Brief explanation of classification'))]),
          simple=1)]),
    FunctionDef(
      name='route_query',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value="\n    Node purpose: Determines the intent of the user's query (e.g., calorie tracking for food/exercise, historical Q&A, net effect Q&A) and routes to the appropriate node.\n    Implementation reasoning: This node needs structured decision making for routing, hence LLM with structured output is used.\n    ")),
        Assign(
          targets=[
            Name(id='llm', ctx=Store())],
          value=Call(
            func=Name(id='ChatOpenAI', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Constant(value='gpt-4o')),
              keyword(
                arg='temperature',
                value=Constant(value=0))])),
        Assign(
          targets=[
            Name(id='structured_llm', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='llm', ctx=Load()),
              attr='with_structured_output',
              ctx=Load()),
            args=[
              Name(id='IntentClassification', ctx=Load())])),
        Assign(
          targets=[
            Name(id='user_message', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='prompt', ctx=Store())],
          value=JoinedStr(
            values=[
              Constant(value='Classify the intent of this user message: '),
              FormattedValue(
                value=Name(id='user_message', ctx=Load()),
                conversion=-1),
              Constant(value=". Choose from 'track_food', 'track_exercise', 'get_diet_history', 'get_exercise_history', 'get_net_effect'.")])),
        Assign(
          targets=[
            Name(id='result', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='structured_llm', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Name(id='prompt', ctx=Load())])),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='intent')],
            values=[
              List(
                elts=[
                  Call(
                    func=Name(id='AIMessage', ctx=Load()),
                    keywords=[
                      keyword(
                        arg='content',
                        value=JoinedStr(
                          values=[
                            Constant(value='Intent classified as: '),
                            FormattedValue(
                              value=Attribute(
                                value=Name(id='result', ctx=Load()),
                                attr='intent',
                                ctx=Load()),
                              conversion=-1)]))])],
                ctx=Load()),
              Attribute(
                value=Name(id='result', ctx=Load()),
                attr='intent',
                ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='track_food_calories_tools', ctx=Store())],
      value=List(
        elts=[
          Name(id='calorie_lookup_tool', ctx=Load()),
          Name(id='store_food_intake_tool', ctx=Load())],
        ctx=Load())),
    FunctionDef(
      name='track_food_calories',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='\n    Node purpose: Calculates and stores calories consumed from user-reported food intake. It extracts food items and quantities from the user input, looks up calorie information, and stores it in the database.\n    Implementation reasoning: This node interacts with external tools (calorie lookup and storage), so a ReAct agent is suitable.\n    ')),
        Assign(
          targets=[
            Name(id='agent', ctx=Store())],
          value=Call(
            func=Name(id='create_react_agent', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Call(
                  func=Name(id='ChatOpenAI', ctx=Load()),
                  keywords=[
                    keyword(
                      arg='model',
                      value=Constant(value='gpt-4o')),
                    keyword(
                      arg='temperature',
                      value=Constant(value=0.7))])),
              keyword(
                arg='tools',
                value=Name(id='track_food_calories_tools', ctx=Load())),
              keyword(
                arg='prompt',
                value=Constant(value="You are an AI assistant that tracks food calorie intake. Extract food items and quantities from the user input, look up their calories using 'calorie_lookup_tool', calculate total calories, and then store each food item with its calories using 'store_food_intake_tool'. Today's date is 2023-10-27. Respond with the total calories and storage status."))])),
        Assign(
          targets=[
            Name(id='user_input', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='response', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='agent', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Dict(
                keys=[
                  Constant(value='messages')],
                values=[
                  List(
                    elts=[
                      Call(
                        func=Name(id='HumanMessage', ctx=Load()),
                        keywords=[
                          keyword(
                            arg='content',
                            value=Name(id='user_input', ctx=Load()))])],
                    ctx=Load())])])),
        Assign(
          targets=[
            Name(id='food_items_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Food Items: (.*?)Total Calories: (\\d+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load()),
              Attribute(
                value=Name(id='re', ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='calculated_calories_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Total Calories: (\\d+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='storage_status_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Storage Status: (\\w+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='food_items', ctx=Store())],
          value=List(ctx=Load())),
        If(
          test=Name(id='food_items_match', ctx=Load()),
          body=[
            Assign(
              targets=[
                Name(id='items_str', ctx=Store())],
              value=Call(
                func=Attribute(
                  value=Call(
                    func=Attribute(
                      value=Name(id='food_items_match', ctx=Load()),
                      attr='group',
                      ctx=Load()),
                    args=[
                      Constant(value=1)]),
                  attr='strip',
                  ctx=Load()))),
            For(
              target=Name(id='line', ctx=Store()),
              iter=Call(
                func=Attribute(
                  value=Name(id='items_str', ctx=Load()),
                  attr='split',
                  ctx=Load()),
                args=[
                  Constant(value='\n')]),
              body=[
                If(
                  test=Compare(
                    left=Constant(value=':'),
                    ops=[
                      In()],
                    comparators=[
                      Name(id='line', ctx=Load())]),
                  body=[
                    Assign(
                      targets=[
                        Name(id='parts', ctx=Store())],
                      value=Call(
                        func=Attribute(
                          value=Name(id='line', ctx=Load()),
                          attr='split',
                          ctx=Load()),
                        args=[
                          Constant(value=':')])),
                    If(
                      test=Compare(
                        left=Call(
                          func=Name(id='len', ctx=Load()),
                          args=[
                            Name(id='parts', ctx=Load())]),
                        ops=[
                          Eq()],
                        comparators=[
                          Constant(value=2)]),
                      body=[
                        Assign(
                          targets=[
                            Name(id='item', ctx=Store())],
                          value=Call(
                            func=Attribute(
                              value=Subscript(
                                value=Name(id='parts', ctx=Load()),
                                slice=Constant(value=0),
                                ctx=Load()),
                              attr='strip',
                              ctx=Load()))),
                        Assign(
                          targets=[
                            Name(id='quantity', ctx=Store())],
                          value=Call(
                            func=Attribute(
                              value=Call(
                                func=Attribute(
                                  value=Call(
                                    func=Attribute(
                                      value=Subscript(
                                        value=Name(id='parts', ctx=Load()),
                                        slice=Constant(value=1),
                                        ctx=Load()),
                                      attr='strip',
                                      ctx=Load())),
                                  attr='replace',
                                  ctx=Load()),
                                args=[
                                  Constant(value='calories'),
                                  Constant(value='')]),
                              attr='strip',
                              ctx=Load()))),
                        Expr(
                          value=Call(
                            func=Attribute(
                              value=Name(id='food_items', ctx=Load()),
                              attr='append',
                              ctx=Load()),
                            args=[
                              Dict(
                                keys=[
                                  Constant(value='item'),
                                  Constant(value='quantity')],
                                values=[
                                  Name(id='item', ctx=Load()),
                                  Name(id='quantity', ctx=Load())])]))])])])]),
        Assign(
          targets=[
            Name(id='calculated_calories', ctx=Store())],
          value=IfExp(
            test=Name(id='calculated_calories_match', ctx=Load()),
            body=Call(
              func=Name(id='int', ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='calculated_calories_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)])]),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='storage_status', ctx=Store())],
          value=IfExp(
            test=Name(id='storage_status_match', ctx=Load()),
            body=Call(
              func=Attribute(
                value=Name(id='storage_status_match', ctx=Load()),
                attr='group',
                ctx=Load()),
              args=[
                Constant(value=1)]),
            orelse=Constant(value='unknown'))),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='food_items'),
              Constant(value='calculated_calories'),
              Constant(value='storage_status')],
            values=[
              List(
                elts=[
                  Subscript(
                    value=Subscript(
                      value=Name(id='response', ctx=Load()),
                      slice=Constant(value='messages'),
                      ctx=Load()),
                    slice=UnaryOp(
                      op=USub(),
                      operand=Constant(value=1)),
                    ctx=Load())],
                ctx=Load()),
              Name(id='food_items', ctx=Load()),
              Name(id='calculated_calories', ctx=Load()),
              Name(id='storage_status', ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='track_exercise_calories_tools', ctx=Store())],
      value=List(
        elts=[
          Name(id='exercise_calorie_estimator_tool', ctx=Load()),
          Name(id='store_exercise_data_tool', ctx=Load())],
        ctx=Load())),
    FunctionDef(
      name='track_exercise_calories',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='\n    Node purpose: Calculates and stores calories burned from user-reported exercise. It extracts exercise type and duration from the user input, estimates calorie expenditure, and stores it in the database.\n    Implementation reasoning: This node interacts with external tools (exercise calorie estimation and storage), so a ReAct agent is suitable.\n    ')),
        Assign(
          targets=[
            Name(id='agent', ctx=Store())],
          value=Call(
            func=Name(id='create_react_agent', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Call(
                  func=Name(id='ChatOpenAI', ctx=Load()),
                  keywords=[
                    keyword(
                      arg='model',
                      value=Constant(value='gpt-4o')),
                    keyword(
                      arg='temperature',
                      value=Constant(value=0.7))])),
              keyword(
                arg='tools',
                value=Name(id='track_exercise_calories_tools', ctx=Load())),
              keyword(
                arg='prompt',
                value=Constant(value="You are an AI assistant that tracks exercise calorie expenditure. Extract exercise type and duration from the user input, estimate calories burned using 'exercise_calorie_estimator_tool', and then store the exercise data using 'store_exercise_data_tool'. Today's date is 2023-10-27. Respond with the estimated calories burned and storage status."))])),
        Assign(
          targets=[
            Name(id='user_input', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='response', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='agent', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Dict(
                keys=[
                  Constant(value='messages')],
                values=[
                  List(
                    elts=[
                      Call(
                        func=Name(id='HumanMessage', ctx=Load()),
                        keywords=[
                          keyword(
                            arg='content',
                            value=Name(id='user_input', ctx=Load()))])],
                    ctx=Load())])])),
        Assign(
          targets=[
            Name(id='exercise_type_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Exercise Type: (.*?)\\n'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='duration_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Duration: (\\d+) minutes'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='estimated_calories_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Estimated Calories Burned: (\\d+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='storage_status_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Storage Status: (\\w+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='exercise_details', ctx=Store())],
          value=Dict()),
        If(
          test=Name(id='exercise_type_match', ctx=Load()),
          body=[
            Assign(
              targets=[
                Subscript(
                  value=Name(id='exercise_details', ctx=Load()),
                  slice=Constant(value='type'),
                  ctx=Store())],
              value=Call(
                func=Attribute(
                  value=Call(
                    func=Attribute(
                      value=Name(id='exercise_type_match', ctx=Load()),
                      attr='group',
                      ctx=Load()),
                    args=[
                      Constant(value=1)]),
                  attr='strip',
                  ctx=Load())))]),
        If(
          test=Name(id='duration_match', ctx=Load()),
          body=[
            Assign(
              targets=[
                Subscript(
                  value=Name(id='exercise_details', ctx=Load()),
                  slice=Constant(value='duration_minutes'),
                  ctx=Store())],
              value=Call(
                func=Name(id='int', ctx=Load()),
                args=[
                  Call(
                    func=Attribute(
                      value=Name(id='duration_match', ctx=Load()),
                      attr='group',
                      ctx=Load()),
                    args=[
                      Constant(value=1)])]))]),
        Assign(
          targets=[
            Name(id='estimated_calories_burned', ctx=Store())],
          value=IfExp(
            test=Name(id='estimated_calories_match', ctx=Load()),
            body=Call(
              func=Name(id='int', ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='estimated_calories_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)])]),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='storage_status', ctx=Store())],
          value=IfExp(
            test=Name(id='storage_status_match', ctx=Load()),
            body=Call(
              func=Attribute(
                value=Name(id='storage_status_match', ctx=Load()),
                attr='group',
                ctx=Load()),
              args=[
                Constant(value=1)]),
            orelse=Constant(value='unknown'))),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='exercise_details'),
              Constant(value='estimated_calories_burned'),
              Constant(value='storage_status')],
            values=[
              List(
                elts=[
                  Subscript(
                    value=Subscript(
                      value=Name(id='response', ctx=Load()),
                      slice=Constant(value='messages'),
                      ctx=Load()),
                    slice=UnaryOp(
                      op=USub(),
                      operand=Constant(value=1)),
                    ctx=Load())],
                ctx=Load()),
              Name(id='exercise_details', ctx=Load()),
              Name(id='estimated_calories_burned', ctx=Load()),
              Name(id='storage_status', ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='get_diet_history_tools', ctx=Store())],
      value=List(
        elts=[
          Name(id='get_food_history_tool', ctx=Load())],
        ctx=Load())),
    FunctionDef(
      name='get_diet_history',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value="\n    Node purpose: Retrieves and summarizes past food consumption data based on user queries (e.g., 'What did I eat yesterday?'). It queries the database for relevant entries.\n    Implementation reasoning: This node interacts with a tool to retrieve historical data, so a ReAct agent is suitable.\n    ")),
        Assign(
          targets=[
            Name(id='agent', ctx=Store())],
          value=Call(
            func=Name(id='create_react_agent', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Call(
                  func=Name(id='ChatOpenAI', ctx=Load()),
                  keywords=[
                    keyword(
                      arg='model',
                      value=Constant(value='gpt-4o')),
                    keyword(
                      arg='temperature',
                      value=Constant(value=0.7))])),
              keyword(
                arg='tools',
                value=Name(id='get_diet_history_tools', ctx=Load())),
              keyword(
                arg='prompt',
                value=Constant(value="You are an AI assistant that retrieves diet history. Extract the date or date range from the user query and use the 'get_food_history_tool' to retrieve the diet history. Summarize the results for the user."))])),
        Assign(
          targets=[
            Name(id='user_input', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='response', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='agent', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Dict(
                keys=[
                  Constant(value='messages')],
                values=[
                  List(
                    elts=[
                      Call(
                        func=Name(id='HumanMessage', ctx=Load()),
                        keywords=[
                          keyword(
                            arg='content',
                            value=Name(id='user_input', ctx=Load()))])],
                    ctx=Load())])])),
        Assign(
          targets=[
            Name(id='query_date_range_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Date Range: (.*?)\\n'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='diet_history_results_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Diet History: (.*)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load()),
              Attribute(
                value=Name(id='re', ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='query_date_range', ctx=Store())],
          value=IfExp(
            test=Name(id='query_date_range_match', ctx=Load()),
            body=Call(
              func=Attribute(
                value=Call(
                  func=Attribute(
                    value=Name(id='query_date_range_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)]),
                attr='strip',
                ctx=Load())),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='diet_history_results', ctx=Store())],
          value=List(ctx=Load())),
        If(
          test=Name(id='diet_history_results_match', ctx=Load()),
          body=[
            Try(
              body=[
                Assign(
                  targets=[
                    Name(id='diet_history_results', ctx=Store())],
                  value=Call(
                    func=Attribute(
                      value=Name(id='json', ctx=Load()),
                      attr='loads',
                      ctx=Load()),
                    args=[
                      Call(
                        func=Attribute(
                          value=Call(
                            func=Attribute(
                              value=Name(id='diet_history_results_match', ctx=Load()),
                              attr='group',
                              ctx=Load()),
                            args=[
                              Constant(value=1)]),
                          attr='strip',
                          ctx=Load()))]))],
              handlers=[
                ExceptHandler(
                  type=Attribute(
                    value=Name(id='json', ctx=Load()),
                    attr='JSONDecodeError',
                    ctx=Load()),
                  body=[
                    Pass()])])]),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='query_date_range'),
              Constant(value='diet_history_results')],
            values=[
              List(
                elts=[
                  Subscript(
                    value=Subscript(
                      value=Name(id='response', ctx=Load()),
                      slice=Constant(value='messages'),
                      ctx=Load()),
                    slice=UnaryOp(
                      op=USub(),
                      operand=Constant(value=1)),
                    ctx=Load())],
                ctx=Load()),
              Name(id='query_date_range', ctx=Load()),
              Name(id='diet_history_results', ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='get_exercise_history_tools', ctx=Store())],
      value=List(
        elts=[
          Name(id='get_exercise_history_tool', ctx=Load())],
        ctx=Load())),
    FunctionDef(
      name='get_exercise_history',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value="\n    Node purpose: Retrieves and summarizes past exercise data based on user queries (e.g., 'How much did I run last week?'). It queries the database for relevant entries.\n    Implementation reasoning: This node interacts with a tool to retrieve historical data, so a ReAct agent is suitable.\n    ")),
        Assign(
          targets=[
            Name(id='agent', ctx=Store())],
          value=Call(
            func=Name(id='create_react_agent', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Call(
                  func=Name(id='ChatOpenAI', ctx=Load()),
                  keywords=[
                    keyword(
                      arg='model',
                      value=Constant(value='gpt-4o')),
                    keyword(
                      arg='temperature',
                      value=Constant(value=0.7))])),
              keyword(
                arg='tools',
                value=Name(id='get_exercise_history_tools', ctx=Load())),
              keyword(
                arg='prompt',
                value=Constant(value="You are an AI assistant that retrieves exercise history. Extract the date or date range from the user query and use the 'get_exercise_history_tool' to retrieve the exercise history. Summarize the results for the user."))])),
        Assign(
          targets=[
            Name(id='user_input', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='response', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='agent', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Dict(
                keys=[
                  Constant(value='messages')],
                values=[
                  List(
                    elts=[
                      Call(
                        func=Name(id='HumanMessage', ctx=Load()),
                        keywords=[
                          keyword(
                            arg='content',
                            value=Name(id='user_input', ctx=Load()))])],
                    ctx=Load())])])),
        Assign(
          targets=[
            Name(id='query_date_range_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Date Range: (.*?)\\n'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='exercise_history_results_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Exercise History: (.*)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load()),
              Attribute(
                value=Name(id='re', ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='query_date_range', ctx=Store())],
          value=IfExp(
            test=Name(id='query_date_range_match', ctx=Load()),
            body=Call(
              func=Attribute(
                value=Call(
                  func=Attribute(
                    value=Name(id='query_date_range_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)]),
                attr='strip',
                ctx=Load())),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='exercise_history_results', ctx=Store())],
          value=List(ctx=Load())),
        If(
          test=Name(id='exercise_history_results_match', ctx=Load()),
          body=[
            Try(
              body=[
                Assign(
                  targets=[
                    Name(id='exercise_history_results', ctx=Store())],
                  value=Call(
                    func=Attribute(
                      value=Name(id='json', ctx=Load()),
                      attr='loads',
                      ctx=Load()),
                    args=[
                      Call(
                        func=Attribute(
                          value=Call(
                            func=Attribute(
                              value=Name(id='exercise_history_results_match', ctx=Load()),
                              attr='group',
                              ctx=Load()),
                            args=[
                              Constant(value=1)]),
                          attr='strip',
                          ctx=Load()))]))],
              handlers=[
                ExceptHandler(
                  type=Attribute(
                    value=Name(id='json', ctx=Load()),
                    attr='JSONDecodeError',
                    ctx=Load()),
                  body=[
                    Pass()])])]),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='query_date_range'),
              Constant(value='exercise_history_results')],
            values=[
              List(
                elts=[
                  Subscript(
                    value=Subscript(
                      value=Name(id='response', ctx=Load()),
                      slice=Constant(value='messages'),
                      ctx=Load()),
                    slice=UnaryOp(
                      op=USub(),
                      operand=Constant(value=1)),
                    ctx=Load())],
                ctx=Load()),
              Name(id='query_date_range', ctx=Load()),
              Name(id='exercise_history_results', ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='get_net_calorie_effect_tools', ctx=Store())],
      value=List(
        elts=[
          Name(id='get_food_history_tool', ctx=Load()),
          Name(id='get_exercise_history_tool', ctx=Load())],
        ctx=Load())),
    FunctionDef(
      name='get_net_calorie_effect',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='\n    Node purpose: Calculates and provides insights into the overall calorie balance (consumed vs. burned) for a specified period. It queries the database for both food and exercise entries.\n    Implementation reasoning: This node requires retrieving data from multiple sources and performing calculations, so a ReAct agent is suitable.\n    ')),
        Assign(
          targets=[
            Name(id='agent', ctx=Store())],
          value=Call(
            func=Name(id='create_react_agent', ctx=Load()),
            keywords=[
              keyword(
                arg='model',
                value=Call(
                  func=Name(id='ChatOpenAI', ctx=Load()),
                  keywords=[
                    keyword(
                      arg='model',
                      value=Constant(value='gpt-4o')),
                    keyword(
                      arg='temperature',
                      value=Constant(value=0.7))])),
              keyword(
                arg='tools',
                value=Name(id='get_net_calorie_effect_tools', ctx=Load())),
              keyword(
                arg='prompt',
                value=Constant(value="You are an AI assistant that calculates net calorie effect. Extract the period from the user query. Use 'get_food_history_tool' and 'get_exercise_history_tool' to retrieve relevant data. Calculate the total calories consumed and burned for the period, and then the net calorie balance. Provide a summary to the user."))])),
        Assign(
          targets=[
            Name(id='user_input', ctx=Store())],
          value=Subscript(
            value=Name(id='state', ctx=Load()),
            slice=Constant(value='user_input'),
            ctx=Load())),
        Assign(
          targets=[
            Name(id='response', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='agent', ctx=Load()),
              attr='invoke',
              ctx=Load()),
            args=[
              Dict(
                keys=[
                  Constant(value='messages')],
                values=[
                  List(
                    elts=[
                      Call(
                        func=Name(id='HumanMessage', ctx=Load()),
                        keywords=[
                          keyword(
                            arg='content',
                            value=Name(id='user_input', ctx=Load()))])],
                    ctx=Load())])])),
        Assign(
          targets=[
            Name(id='query_period_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Period: (.*?)\\n'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='total_consumed_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Total Calories Consumed: (\\d+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='total_burned_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Total Calories Burned: (\\d+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='net_balance_match', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='re', ctx=Load()),
              attr='search',
              ctx=Load()),
            args=[
              Constant(value='Net Calorie Balance: ([\\d-]+)'),
              Attribute(
                value=Subscript(
                  value=Subscript(
                    value=Name(id='response', ctx=Load()),
                    slice=Constant(value='messages'),
                    ctx=Load()),
                  slice=UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)),
                  ctx=Load()),
                attr='content',
                ctx=Load())])),
        Assign(
          targets=[
            Name(id='query_period', ctx=Store())],
          value=IfExp(
            test=Name(id='query_period_match', ctx=Load()),
            body=Call(
              func=Attribute(
                value=Call(
                  func=Attribute(
                    value=Name(id='query_period_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)]),
                attr='strip',
                ctx=Load())),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='total_calories_consumed', ctx=Store())],
          value=IfExp(
            test=Name(id='total_consumed_match', ctx=Load()),
            body=Call(
              func=Name(id='int', ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='total_consumed_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)])]),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='total_calories_burned', ctx=Store())],
          value=IfExp(
            test=Name(id='total_burned_match', ctx=Load()),
            body=Call(
              func=Name(id='int', ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='total_burned_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)])]),
            orelse=Constant(value=None))),
        Assign(
          targets=[
            Name(id='net_calorie_balance', ctx=Store())],
          value=IfExp(
            test=Name(id='net_balance_match', ctx=Load()),
            body=Call(
              func=Name(id='int', ctx=Load()),
              args=[
                Call(
                  func=Attribute(
                    value=Name(id='net_balance_match', ctx=Load()),
                    attr='group',
                    ctx=Load()),
                  args=[
                    Constant(value=1)])]),
            orelse=Constant(value=None))),
        Return(
          value=Dict(
            keys=[
              Constant(value='messages'),
              Constant(value='query_period'),
              Constant(value='total_calories_consumed'),
              Constant(value='total_calories_burned'),
              Constant(value='net_calorie_balance')],
            values=[
              List(
                elts=[
                  Subscript(
                    value=Subscript(
                      value=Name(id='response', ctx=Load()),
                      slice=Constant(value='messages'),
                      ctx=Load()),
                    slice=UnaryOp(
                      op=USub(),
                      operand=Constant(value=1)),
                    ctx=Load())],
                ctx=Load()),
              Name(id='query_period', ctx=Load()),
              Name(id='total_calories_consumed', ctx=Load()),
              Name(id='total_calories_burned', ctx=Load()),
              Name(id='net_calorie_balance', ctx=Load())]))],
      returns=Name(id='GraphState', ctx=Load())),
    Assign(
      targets=[
        Name(id='workflow', ctx=Store())],
      value=Call(
        func=Name(id='StateGraph', ctx=Load()),
        args=[
          Name(id='GraphState', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='route_query'),
          Name(id='route_query', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='track_food_calories'),
          Name(id='track_food_calories', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='track_exercise_calories'),
          Name(id='track_exercise_calories', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='get_diet_history'),
          Name(id='get_diet_history', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='get_exercise_history'),
          Name(id='get_exercise_history', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_node',
          ctx=Load()),
        args=[
          Constant(value='get_net_calorie_effect'),
          Name(id='get_net_calorie_effect', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Name(id='START', ctx=Load()),
          Constant(value='route_query')])),
    FunctionDef(
      name='route_to_next_node',
      args=arguments(
        args=[
          arg(
            arg='state',
            annotation=Name(id='GraphState', ctx=Load()))]),
      body=[
        Expr(
          value=Constant(value='\n    Routing function for route_query node.\n    Determines the next node based on the identified intent.\n    ')),
        If(
          test=Compare(
            left=Subscript(
              value=Name(id='state', ctx=Load()),
              slice=Constant(value='intent'),
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='track_food')]),
          body=[
            Return(
              value=Constant(value='track_food_calories'))],
          orelse=[
            If(
              test=Compare(
                left=Subscript(
                  value=Name(id='state', ctx=Load()),
                  slice=Constant(value='intent'),
                  ctx=Load()),
                ops=[
                  Eq()],
                comparators=[
                  Constant(value='track_exercise')]),
              body=[
                Return(
                  value=Constant(value='track_exercise_calories'))],
              orelse=[
                If(
                  test=Compare(
                    left=Subscript(
                      value=Name(id='state', ctx=Load()),
                      slice=Constant(value='intent'),
                      ctx=Load()),
                    ops=[
                      Eq()],
                    comparators=[
                      Constant(value='get_diet_history')]),
                  body=[
                    Return(
                      value=Constant(value='get_diet_history'))],
                  orelse=[
                    If(
                      test=Compare(
                        left=Subscript(
                          value=Name(id='state', ctx=Load()),
                          slice=Constant(value='intent'),
                          ctx=Load()),
                        ops=[
                          Eq()],
                        comparators=[
                          Constant(value='get_exercise_history')]),
                      body=[
                        Return(
                          value=Constant(value='get_exercise_history'))],
                      orelse=[
                        If(
                          test=Compare(
                            left=Subscript(
                              value=Name(id='state', ctx=Load()),
                              slice=Constant(value='intent'),
                              ctx=Load()),
                            ops=[
                              Eq()],
                            comparators=[
                              Constant(value='get_net_effect')]),
                          body=[
                            Return(
                              value=Constant(value='get_net_calorie_effect'))],
                          orelse=[
                            Return(
                              value=Name(id='END', ctx=Load()))])])])])])],
      returns=Name(id='str', ctx=Load())),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_conditional_edges',
          ctx=Load()),
        args=[
          Constant(value='route_query'),
          Name(id='route_to_next_node', ctx=Load()),
          Dict(
            keys=[
              Constant(value='track_food'),
              Constant(value='track_exercise'),
              Constant(value='get_diet_history'),
              Constant(value='get_exercise_history'),
              Constant(value='get_net_effect')],
            values=[
              Constant(value='track_food_calories'),
              Constant(value='track_exercise_calories'),
              Constant(value='get_diet_history'),
              Constant(value='get_exercise_history'),
              Constant(value='get_net_calorie_effect')])])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Constant(value='track_food_calories'),
          Name(id='END', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Constant(value='track_exercise_calories'),
          Name(id='END', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Constant(value='get_diet_history'),
          Name(id='END', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Constant(value='get_exercise_history'),
          Name(id='END', ctx=Load())])),
    Expr(
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='add_edge',
          ctx=Load()),
        args=[
          Constant(value='get_net_calorie_effect'),
          Name(id='END', ctx=Load())])),
    Assign(
      targets=[
        Name(id='checkpointer', ctx=Store())],
      value=Call(
        func=Name(id='InMemorySaver', ctx=Load()))),
    Assign(
      targets=[
        Name(id='app', ctx=Store())],
      value=Call(
        func=Attribute(
          value=Name(id='workflow', ctx=Load()),
          attr='compile',
          ctx=Load()),
        keywords=[
          keyword(
            arg='checkpointer',
            value=Name(id='checkpointer', ctx=Load()))]))])