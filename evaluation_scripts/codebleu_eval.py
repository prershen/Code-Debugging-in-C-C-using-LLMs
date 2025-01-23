from codebleu import calc_codebleu

prediction = "def add ( a , b ) :\n return a + b"
reference = "def sum ( first , second ) :\n return second + first"

prediction_c = ["""
#include<stdio.h>
    int main()
    {
        int a,d;
        scanf ("%d",&a);
        printf ("Reverse of %d is ",a);
        while (a!=0){
            d=a%10;
            if (d==0)break;
            else printf ("%d",d)
            
            a=a/10;
        }
        return 0;
}
                """,
"""
#include<stdio.h>
    int main()
    {
        int a,d;
        scanf ("%d",&a);
        printf ("Reverse of %d is ",a);
        while (a!=0){
            d=a%10;
            if (d==0)break;
            else printf ("%d",d)
            
            a=a/10;
        }
        return 0;
}

""",
"""
#include<stdio.h>
    int main()
    {
        int a,d;
        scanf ("%d",&a);
        printf ("Reverse of %d is ",a);
        while (a!=0){
            d=a%10;
            if (d==0)break;
            else printf ("%d",d)
            
            a=a/10;
        }
        return 0;
}
""",
"""
#include<stdio.h>
    int main()
    {
        int a,d;
        scanf ("%d",&a);
        printf ("Reverse of %d is ",a);
        while (a!=0){
            d=a%10;
            if (d==0)break;
            else printf ("%d",d)
            
            a=a/10;
        }
        return 0;
}
""",
"""
#include<stdio.h>
int main()
{
    int a,d;
    scanf ("%d",&a);
    printf ("Reverse of %d is ",a);
    while (a!=0){
        d=a%10;
        if (d==0)break;
        else printf ("%d",d)
        
        a=a/10;
    }
    return 0;
}

""",
"""
#include<stdio.h>
int main()
{
    int a,d;
    scanf ("%d",&a);
    printf ("Reverse of %d is ",a);
    while (a!=0){
        d=a%10;
        if (d==0)break;
        else printf ("%d",d)
        
        a=a/10;
    }
    return 0;
}
""",
"""
#include<stdio.h>
int main()
{
    int a,d;
    scanf ("%d",&a);s
    printf ("Reverse of %d is ",a);
    while (a!=0){
        d=a%10;
        if (d==0)break;
        else printf ("%d",d)
        
        a=a/10;
    }
    return 0;
}
"""
]

reference_c = """
#include<stdio.h>

int main()
{
    int a,d;
    scanf ("%d",&a);
    printf ("Reverse of %d is ",a);
    while (a!=0){
        d=a%10;
        if (d==0)break;
        else printf ("%d",d);
        
        a=a/10;
    }  
    return 0;
}
"""

prediction_java = [
"""
public static double mean_absolute_deviation(List<Double> numbers) {
    double sum = 0.;
    for (Double number : numbers){
        sum += Math.abs(number);
    }
    return sum / numbers.size();
}
""",
"""
public static double mean_absolute_deviation(List<Double> numbers) {
    double sum = 0.;
    for (Double number : numbers){
        sum += Math.abs(number);
    }
    return sum / numbers.size();
}
"""
]
reference_java = """
public static double mean_absolute_deviation(List<Double> numbers) {
    double sum = 0.0;
    for (Double number : numbers) {
        sum += number;
    }
    double mean = sum / numbers.size();
    double deviation_sum = 0.0;
    for (Double number : numbers) {
        deviation_sum += Math.abs(number - mean);
    }
    return deviation_sum / numbers.size();
}
"""
maxres = 0
for i,pred in enumerate(prediction_c):
    print("Prediction: ", i)
    result = calc_codebleu([reference_c], [pred], lang="cpp", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    print(result)
    maxres = max(maxres, result['codebleu'])
print("Max CodeBLEU Prediction: ", maxres)
# import evaluate
# metric = evaluate.load("dvitel/codebleu")

# prediction = "def add ( a , b ) :\n return a + b"
# reference = "def sum ( first , second ) :\n return second + first"

# result = metric.compute([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25))