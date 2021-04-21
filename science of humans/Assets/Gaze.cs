using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Gaze : MonoBehaviour
{
    public Transform leftEye;
    public Transform rightEye;
    public Transform target;
    // Angular speed in radians per sec.
    public float speed = 1.0f;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void LateUpdate()
    {
        // Determine which direction to rotate towards
        Vector3 leftEyeDirection = target.position - leftEye.position;
        // Determine which direction to rotate towards
        Vector3 rightEyeDirection = target.position - rightEye.position;

        // The step size is equal to speed times frame time.
        float singleStep = speed * Time.deltaTime;

        // Rotate the forward vector towards the target direction by one step
        Vector3 newLeftDirection = Vector3.RotateTowards(leftEye.forward, leftEyeDirection, 1, 0.0f);
        // Rotate the forward vector towards the target direction by one step
        Vector3 newRightDirection = Vector3.RotateTowards(rightEye.forward, rightEyeDirection, 1, 0.0f);

        // Draw a ray pointing at our target in
        //Debug.DrawRay(transform.position, newDirection, Color.red);

        // Calculate a rotation a step closer to the target and applies rotation to this object
        leftEye.rotation = Quaternion.LookRotation(newLeftDirection);
        // Calculate a rotation a step closer to the target and applies rotation to this object
        rightEye.rotation = Quaternion.LookRotation(newRightDirection);
    }
}
