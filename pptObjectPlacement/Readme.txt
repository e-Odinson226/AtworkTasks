12/04/1401

# Problem:
At this time agent doesn't calculate coordinates to put down objects, on the platform.


# Solution
so the goal is:

1.  Have a FOV of the platform (which, objects are going to be put on),
    the robot should place objects according to their dimensions and platform's available areas.
    
    STATE: developing
        
        
2.  Have an algorithm to arrange objects in such a way,
    to be the optimum state of placing objects according to proportions.(like a puzzle)
    
    STATE: 

note:
    1. all functions have a document part (written as comments) above them that exhibit what's their task and how they get it done.
    2. documentation structure:
        TAKES(function inputs) [...]
        DOES(what this function does) [...]
        RETURN(function output) [...]

        each part notation is represented in this way:
        x-ly_lz         ---->       [x]: task number,
                                    [y]: beginning line,
                                    [z]: finishing line.