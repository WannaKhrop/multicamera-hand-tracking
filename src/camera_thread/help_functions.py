# just algorithms that will be used in project
import numpy as np

def merge_sorted_lists(list1: list[tuple[int, np.array]],
                       list2: list[tuple[int, np.array]],
                       source1: int,
                       source2: int
                       ) -> list[tuple[int, int, np.array]]:
    
    i, j = 0, 0
    merged_list = []
    
    # Loop until one of the lists is exhausted
    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            merged_list.append((list1[i][0], source1, list[i][1]))
            i += 1
        else:
            merged_list.append((list2[j][0], source2, list[j][1]))
            j += 1
    
    # Append remaining elements from list1 if any
    while i < len(list1):
        merged_list.append((list1[i][0], source1, list[i][1]))
        i += 1
    
    # Append remaining elements from list2 if any
    while j < len(list2):
        merged_list.append((list2[j][0], source2, list[j][1]))
        j += 1

    return merged_list