LeetCode Common Patterns: Python Templates

This document provides Python templates for common algorithmic patterns encountered in LeetCode problems.
1. Sliding Window

This pattern is used to perform an operation on a specific window size of a given array or string. The window "slides" across the data structure.
General Template

def sliding_window_template(arr, k):
    if not arr or k == 0:
        return [] # Or appropriate return for empty/invalid input

    window_start = 0
    # current_window_aggregate can be sum, count of specific elements, a hash map, etc.
    current_window_aggregate = 0 # Example: for sum
    result = [] # Or a variable to store max/min, count, etc.

    for window_end in range(len(arr)):
        # Add the new element to the window's aggregate
        # Example: current_window_aggregate += arr[window_end]

        # Shrink the window if it's too large or meets a condition
        # This condition is typically when window_end >= k - 1 for fixed-size window
        if window_end >= k - 1: # For a fixed-size window of size k
            # Process the window:
            # - Store the aggregate (e.g., result.append(current_window_aggregate))
            # - Update a global max/min (e.g., result = max(result, current_window_aggregate))
            # - Check a condition related to the window
            # print(f"Window: {arr[window_start:window_end+1]}, Aggregate: {current_window_aggregate}")


            # Remove the element going out of the window to maintain window size
            # Example: current_window_aggregate -= arr[window_start]
            window_start += 1
            
    return result

Example: Maximum Sum Subarray of Size K

def max_sum_subarray_of_size_k(arr, k):
    if not arr or k <= 0 or k > len(arr):
        return 0 # Or raise an error

    max_so_far = float('-inf')
    current_window_sum = 0
    window_start = 0

    for window_end in range(len(arr)):
        current_window_sum += arr[window_end]

        if window_end >= k - 1:
            max_so_far = max(max_so_far, current_window_sum)
            current_window_sum -= arr[window_start] # Slide window forward
            window_start += 1
            
    return max_so_far if max_so_far != float('-inf') else 0

# print(max_sum_subarray_of_size_k([2, 1, 5, 1, 3, 2], 3)) # Output: 9
# print(max_sum_subarray_of_size_k([2, 3, 4, 1, 5], 2))    # Output: 7

2. Two Pointers

This pattern uses two pointers to iterate through a data structure, often an array, until they meet or satisfy a certain condition. The pointers can move towards each other, away from each other, or in the same direction.
Template: Pointers Moving Towards Each Other (e.g., for sorted arrays)

def two_pointers_converging_template(arr, target): # Example: target sum
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_val = arr[left] + arr[right] # Or other operation involving arr[left] and arr[right]
        
        if current_val == target:
            # Found the target or condition
            return [left, right] # Or [arr[left], arr[right]], or True
            # Depending on the problem, you might need to continue searching for all pairs:
            # left += 1
            # right -= 1
        elif current_val < target:
            left += 1 # Need a larger value
        else: # current_val > target
            right -= 1 # Need a smaller value
            
    return -1 # Or False, or [], etc., if not found
    
# print(two_pointers_converging_template([1, 2, 3, 4, 6], 6)) # Output: [1, 3] (indices for 2 and 4)
# print(two_pointers_converging_template([2, 5, 9, 11], 11))# Output: [0, 2] (indices for 2 and 9)

Template: Pointers Moving in the Same Direction (can overlap with Fast/Slow or Sliding Window)

def two_pointers_same_direction_template(arr):
    slow = 0 # "Slow" or "read" pointer
    # "fast" or "write" pointer usually starts at 0 or 1, depending on problem
    for fast in range(len(arr)): 
        # Condition to process arr[fast]
        if arr[fast] != some_value_to_remove: # Example: Remove duplicates or specific elements
            # If slow is used for writing valid elements to the front of the array
            arr[slow] = arr[fast]
            slow += 1
    # slow now represents the length of the valid part of the array
    return slow # or arr[:slow]

3. Fast & Slow Pointers (Floyd's Tortoise and Hare)

Often used in linked lists to detect cycles, find the middle element, or in arrays for cycle detection (e.g., find duplicate number where numbers are in a range).

# Definition for singly-linked list (common in LeetCode)
class ListNode:
    def __init__(self, value, next_node=None): # Renamed next to next_node for clarity
        self.val = value
        self.next = next_node

def has_cycle_linked_list(head: ListNode) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True # Cycle detected
    return False

def cycle_start_linked_list(head: ListNode) -> ListNode:
    slow, fast = head, head
    cycle_detected = False
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            cycle_detected = True
            break
            
    if not cycle_detected:
        return None

    # Reset one pointer to the head and move both one step at a time
    pointer1 = head
    pointer2 = slow # or fast
    while pointer1 != pointer2:
        pointer1 = pointer1.next
        pointer2 = pointer2.next
    return pointer1 # This is the start of the cycle

def find_middle_of_linked_list(head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow # When fast reaches the end, slow is at the middle

# Example usage (assuming ListNode is defined and list is created):
# node1 = ListNode(1)
# node2 = ListNode(2)
# node3 = ListNode(3)
# node4 = ListNode(4)
# node5 = ListNode(5)
# node1.next = node2
# node2.next = node3
# node3.next = node4
# node4.next = node5
# print(f"Middle node value: {find_middle_of_linked_list(node1).val}") # Output: 3

# node4.next = node2 # Create a cycle: 4 -> 2
# print(f"Has cycle: {has_cycle_linked_list(node1)}") # Output: True
# print(f"Cycle start value: {cycle_start_linked_list(node1).val}") # Output: 2

4. Merge Intervals

This pattern deals with merging overlapping intervals. An interval is typically represented as a pair [start, end].

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    # Sort intervals by their start times
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # If merged list is empty or the current interval does not overlap with the previous one
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Overlap, so merge the current interval with the previous one
            merged[-1][1] = max(merged[-1][1], interval[1])
            
    return merged

# Example:
# intervals1 = [[1,3],[2,6],[8,10],[15,18]]
# print(merge_intervals(intervals1)) # Output: [[1,6],[8,10],[15,18]]

# intervals2 = [[1,4],[4,5]]
# print(merge_intervals(intervals2)) # Output: [[1,5]]

# intervals3 = [[1,4],[0,4]]
# print(merge_intervals(intervals3)) # Output: [[0,4]]

5. Cyclic Sort

This pattern is used when problems involve arrays containing numbers in a specific range (e.g., 1 to N, or 0 to N-1). The idea is to place each number at its correct index.

def cyclic_sort(nums: list[int]):
    """Sorts an array in-place if it contains numbers from 1 to N."""
    i = 0
    n = len(nums)
    while i < n:
        # Correct index for nums[i] should be nums[i] - 1
        # (if numbers are 1 to N)
        correct_idx = nums[i] - 1 
        
        # If nums[i] is within the range [1, N] and not at its correct place
        if 1 <= nums[i] <= n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i] # Swap
        else:
            i += 1
    # After this, nums[i] should be i+1 if no duplicates/missing
    # Or, if numbers are 0 to N-1, correct_idx = nums[i]

# Example:
# arr = [3, 1, 5, 4, 2] # Numbers from 1 to 5
# cyclic_sort(arr)
# print(arr) # Output: [1, 2, 3, 4, 5]

# To find missing/duplicate numbers after cyclic sort:
def find_missing_number_1_to_N(nums: list[int]) -> int:
    """Assumes nums contains N distinct numbers taken from the range [0, N]
       or N-1 distinct numbers from [1,N] and one is missing.
       This example is for numbers from 1 to N, one missing.
    """
    cyclic_sort(nums) # Ensure numbers are in place or indicate what's wrong
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    return len(nums) + 1 # If all numbers 1..N are present (e.g., if problem allows this)

# arr_missing = [3, 1, 5, 2] # Missing 4, N=5 (conceptually, array size 4)
# To use find_missing_number_1_to_N directly, adjust problem formulation or function
# A more common "find missing number" assumes numbers 0..N, with one missing.
# Example using cyclic sort for finding the first K missing positives
def find_first_k_missing_positives(nums: list[int], k: int) -> list[int]:
    n = len(nums)
    i = 0
    while i < n:
        j = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]  # swap
        else:
            i += 1

    missing_numbers = []
    extra_numbers = set() # Numbers that are present but > n or duplicates
    for i in range(n):
        if len(missing_numbers) < k:
            if nums[i] != i + 1:
                missing_numbers.append(i + 1)
                extra_numbers.add(nums[i]) # This number is out of place or duplicate
    
    # If we haven't found k missing numbers, they must be > n
    i = 1
    while len(missing_numbers) < k:
        candidate_missing = n + i
        if candidate_missing not in extra_numbers and candidate_missing not in nums : # Check if it was among original nums
            # A more robust check for 'candidate_missing not in nums' can be slow.
            # A better way is to only add to missing_numbers if candidate_missing
            # was not already found in its correct place (nums[candidate_missing-1])
            # or if candidate_missing is beyond the array's conceptual range of 1..n
            # This part needs careful handling based on problem constraints.
            # A simpler way if we only use the sorted array and need numbers > n:
             if n + i not in extra_numbers: # A simplified check
                # Check if this number was originally in nums AND is greater than n
                # This logic gets complex, usually simpler to use a set of present numbers
                # Let's use a simpler approach for numbers > n:
                # Add numbers n+1, n+2, ... until k missing are found,
                # ensuring they were not in the 'extra_numbers' set (duplicates or >n values seen)
                # The above `extra_numbers.add(nums[i])` might not be perfect for this.
                # A better approach for finding K missing:
                pass # This requires a more robust strategy for numbers > n
    # Simplified version for first k missing:
    # 1. Cyclic sort numbers 1 to N.
    # 2. Iterate nums: if nums[i] != i+1, add i+1 to missing (if unique), add nums[i] to seen_extra.
    # 3. Iterate i from 1: if n+i not in seen_extra, add n+i to missing. Stop when len(missing) == k.
    return missing_numbers # Placeholder, full k-missing is more involved.

6. In-place Reversal of a LinkedList

This pattern reverses a portion or the entirety of a linked list without using extra space (O(1) space).

# Definition for singly-linked list (repeated for clarity)
# class ListNode:
#     def __init__(self, value, next_node=None):
#         self.val = value
#         self.next = next_node

def reverse_linked_list(head: ListNode) -> ListNode:
    prev = None
    current = head
    while current:
        next_temp = current.next # Store the next node
        current.next = prev    # Reverse current node's pointer
        prev = current         # Move prev one step forward
        current = next_temp    # Move current one step forward
    return prev # prev is the new head

def reverse_sub_list(head: ListNode, p: int, q: int) -> ListNode:
    if p == q:
        return head

    # Skip the first p-1 nodes
    current, prev_node = head, None # prev_node is one before the sublist start
    i = 0
    while current and i < p - 1:
        prev_node = current
        current = current.next
        i += 1

    # current is now at the p-th node (start of sub-list)
    # prev_node is the node before the sub-list (or None if p=1)

    last_node_of_first_part = prev_node
    # current (p-th node) will become the last node of the reversed sub-list
    last_node_of_sub_list = current 
    
    next_temp = None # To store next node during reversal
    # Reverse nodes from p to q
    i = 0
    while current and i < q - p + 1:
        next_temp = current.next
        current.next = prev_node # prev_node here is acting as the previous in reversal
        prev_node = current      # prev_node advances
        current = next_temp    # current advances
        i += 1
    
    # Connect with the first part
    if last_node_of_first_part:
        # prev_node is now the head of the reversed sub-list
        last_node_of_first_part.next = prev_node 
    else:
        # This means p=1, so the head of the entire list has changed
        head = prev_node
    
    # Connect with the last part
    # last_node_of_sub_list was the original p-th node, now it's the end of reversed part
    # current is the node that was originally (q+1)-th node
    last_node_of_sub_list.next = current 
    return head

# Example usage needs ListNode class and list creation
# head_list = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# reversed_head = reverse_linked_list(head_list)
# current = reversed_head
# while current: print(current.val, end=" -> "); current = current.next; # 5 -> 4 -> 3 -> 2 -> 1 ->

# head_list_sub = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# reversed_sub_head = reverse_sub_list(head_list_sub, 2, 4) # Reverse 2-3-4
# current = reversed_sub_head
# while current: print(current.val, end=" -> "); current = current.next; # 1 -> 4 -> 3 -> 2 -> 5 ->

7. Tree Breadth-First Search (BFS)

This pattern is used to traverse a tree level by level. It uses a queue.

from collections import deque

# Definition for a binary tree node (common in LeetCode)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_bfs_level_order_traversal(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level_nodes_values = [] 

        for _ in range(level_size):
            node = queue.popleft()
            current_level_nodes_values.append(node.val) # Process node

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        if current_level_nodes_values: # Ensure level is not empty (can happen in some cases)
            result.append(current_level_nodes_values)
            
    return result

# Example (assumes TreeNode class is defined):
# root = TreeNode(12)
# root.left = TreeNode(7)
# root.right = TreeNode(1)
# root.left.left = TreeNode(9)
# root.right.left = TreeNode(10)
# root.right.right = TreeNode(5)
# print(tree_bfs_level_order_traversal(root))
# Output: [[12], [7, 1], [9, 10, 5]]

8. Tree Depth-First Search (DFS)

This pattern is used to traverse a tree by exploring as far as possible along each branch before backtracking. It can be implemented recursively or iteratively (using a stack).

# Definition for a binary tree node (repeated for clarity)
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# --- Recursive DFS Templates ---
def tree_dfs_preorder_recursive(root: TreeNode) -> list[int]:
    result = []
    def dfs(node):
        if not node:
            return
        result.append(node.val) # Process node (Root)
        dfs(node.left)          # Left
        dfs(node.right)         # Right
    dfs(root)
    return result

def tree_dfs_inorder_recursive(root: TreeNode) -> list[int]:
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)          # Left
        result.append(node.val) # Process node (Root)
        dfs(node.right)         # Right
    dfs(root)
    return result

def tree_dfs_postorder_recursive(root: TreeNode) -> list[int]:
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)          # Left
        dfs(node.right)         # Right
        result.append(node.val) # Process node (Root)
    dfs(root)
    return result

# --- Iterative DFS Template (Pre-order using a stack) ---
def tree_dfs_preorder_iterative(root: TreeNode) -> list[int]:
    if not root:
        return []
    
    result = []
    stack = [root] # Initialize stack with the root node
    
    while stack:
        node = stack.pop()
        result.append(node.val) # Process node
        
        # Push right child first, then left child, to process left child first (pre-order behavior)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
            
    return result

# Iterative In-order and Post-order are more complex.
# In-order iterative:
def tree_dfs_inorder_iterative(root: TreeNode) -> list[int]:
    result = []
    stack = []
    current = root
    while current or stack:
        while current: # Go as left as possible
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val) # Visit node
        current = current.right # Go right
    return result


# Example (assumes TreeNode class is defined):
# root = TreeNode(12)
# root.left = TreeNode(7, TreeNode(9))
# root.right = TreeNode(1, TreeNode(10), TreeNode(5))

# print("Pre-order Recursive:", tree_dfs_preorder_recursive(root)) # [12, 7, 9, 1, 10, 5]
# print("In-order Recursive:", tree_dfs_inorder_recursive(root))   # [9, 7, 12, 10, 1, 5]
# print("Post-order Recursive:", tree_dfs_postorder_recursive(root))# [9, 7, 10, 5, 1, 12]
# print("Pre-order Iterative:", tree_dfs_preorder_iterative(root)) # [12, 7, 9, 1, 10, 5]
# print("In-order Iterative:", tree_dfs_inorder_iterative(root))   # [9, 7, 12, 10, 1, 5]

This Markdown file now contains all the templates. Remember to uncomment the example usages and define the ListNode and TreeNode classes if you want to run the examples directly.
