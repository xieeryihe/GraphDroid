class AgentPrompt:
    ActionGenerator = """
    You are a front-end testing engineer. You need to interact with a smartphone app to accomplish task "{task}".
    I will show you the page image with nodes information and action history, please determine which action should you choose.
    
    <Nodes ID>
    {element_info}
    </Nodes ID>
    <action history>
    {action_history}
    </action history>

    {addition_utg}

    Please respond strictly in a JSON format:
    1.click:
    ```json
    {{
        "action": "click",
        "node_id": "node ID from <Nodes ID>",
        "reason": "why you're clicking this" 
    }}
    ```
    2.back:
    ```json
    {{
        "action": "back",
        "reason": "why you need to go back"
    }}
    ```
    3.stop:
    ```json
    {{
        "action": "stop",
        "reason": "why you think the task is completed"
    }}
    ```
    <Note>
    1. If you have reached the target page or completed the task, don't do any other actions, just do 'stop'.
    2. If you need to go back to the previous page, do 'back' action.
    3. If there is a pop-window in the page, first try to close it by clicking the 'X', 'I know' or 'close' button.
    4. You should call one action per response.
    </Note>
    """
        
    ActionGenerator2 = """
    You are a front-end testing engineer. You need to interact with a smartphone app to accomplish task "{task}".
    I will show you the page image with nodes information and action history, please determine which action should you choose.
    
    <Nodes ID>
    {element_info}
    </Nodes ID>
    <action history>
    {action_history}
    </action history>

    {addition_utg}

    Please respond strictly in a JSON format:
    1.click:
    ```json
    {{
        "action": "click",
        "node_id": "node ID from <Nodes ID>",
        "reason": "why you're clicking this" 
    }}
    ```
    2.back:
    ```json
    {{
        "action": "back",
        "reason": "why you need to go back"
    }}
    ```
    3.input:
    ```json
    {{
        "action": "input",
        "text": "text to enter",
        "reason": "why this text is needed"
    }}
    ```
    4.scroll:
    ```json
    {{
        "action": "scroll",
        "node_id": "node ID from <Nodes ID>",
        "direction": "[up, down, right, left]",
        "reason": "why swiping is necessary"
    }}
    ```
    5.stop:
    ```json
    {{
        "action": "stop",
        "reason": "why you think the task is completed"
    }}
    ```
    <Note>
    1. If you have reached the target page or completed the task, don't do any other actions, just do 'stop'.
    2. If you need to go back to the previous page, do 'back' action.
    3. You should call one action per response.
    4. If you need input text, do 'click' before 'input' to select the input box (if the input box is not selected).
    </Note>
    """

    # 额外添加到 ActionGenerator 中的部分, 需要先构造好 UTG, 再构造 ActionGenerator
    AdditionUTG = """
    Here is the path chain that you can refer to when making decisions.
    It may contain important pre_actions leading to the target page, so please try to follow it.
    <Path Chain>
    {path_chain}
    </Path Chain>
    """

    PageMapping = """
    You are a front-end testing engineer. You need to interact with a smartphone app to accomplish <task>.
    <task>
    {task}
    </task>
    I will show you the page image of the current page.
    Your task is to find the "current page id" and  "target page id" of the <task> in the UI Transition Graph (UTG).
    If task includes multiple pages, use the first page that you can perform the task on as the "target page".
    Becasue the task may not start from the home page, you need to find the target page that you can perform the task on.
    Here is the UTG that you can refer to when making decisions.
    Each node represents a unique UI state (page), and each directed edge represents a user action that transitions from one UI state to another.
    <UTG>
    {utg}
    </UTG>

    Please respond strictly in a JSON format:
    ```json
    {{
        "current_page_id": "the current page of the task, '-1' if not found",
        "target_page_id": "the target page of the task, '-1' if not found",
        "reason": "why you think this page is the target page of the <task>"
    }}
    ```
    """
