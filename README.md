# Transformers
Everything you should know about Transformers

Any NLP problem basic requirement is how to convert words to numbers?

**Vectorization**

Converting words to numbers/vectors (**Vectorization**)

Several methods to convert words to numbers, like

- One Hot Encoding

- Bag of Words (BoW)

- TF-IDF

- Word Embeddings => Captures **semantic meaning** of the word

## **Word Embeddings**

It is nothing but an **n-dim vector**

Eg: King & Queen are similar vectors of dimension n (let's say 300), where each dimension refers to some feature like roaylty, gender, etc

Word Embeddings have a problem of **Average Meaning**

Apple => could be fruit or tech company

Let's say we have 10K sentences and 9K of them are regarding fruit and remaining 1K about the tech company

And based on that let the overall vector (assume only 2 features/dim.) = [0.9     0.3]

Where 0.9 refers to fruit and 0.3 to tech company

New test example: Apple launched a new phone while I was eating the orange.

As per our word embeddings, the Apple here refers to fruit as the overall vector has high value (0.9) for fruit aspect. But clearly it's about tech company.

## **Contextual Embeddings**

It is based on the context, and unlike Word embedding which is static, Contextual embeddings are **dynamic**

They are smart enough to identify that the above new example refers to tech company even though the sentence contains orange (fruit) which doesn't alter it's value as it is context based.

## **Self-Attention Mechanism**

Converts Static Word Embeddings to Dynamic Smart Contextual Embeddings

S1: Money Bank Grows

S2: River Bank Flows

Each **bank** has different meaning here

**Similarity Scores**:

In S1, bank depends on Money and Grows (context). Likely in S2, bank depends on River and Flows which are represented using the similarity scores.

Emoney(new) = 0.7*Emoney + 0.2*Ebank + 0.1*Egrows

Ebank(new) = 0.25*Emoney + 0.7*Ebank + 0.05*Egrows

Egrows(new) = 0.1*Emoney + 0.2*Ebank + 0.7*Egrows

The reason we do **dot product** is because it **quantifies the similarity**

**Dynamic Smart Contextual Embeddings (Y)**

Ymoney = w11*Emoney + w12*Ebank + w13*Egrows

Ybank = w21*Emoney + w22*Ebank + w23*Egrows

Ygrows = w31*Emoney + w32*Ebank + w33*Egrows

Where Wij = Softmax(Sij)

W21 = e^S21/(e^S21 + e^S22 + e^S23)

**Query, Key and Value (Q, K & V) Vectors:**

Thus, W = Softmax(Q*K) and Y = W*V

We can **parallelize** this entire process without losing the order of the sentence (**Positional Encoding**). Thus faster training can be achieved.

No learning parameters involved, no learning happening here

If it's a **Task Specific Contextual Embedding** => Introduce weights and bias so that it can learn from the data

Bank => Ebank => Qbank, Kbank, Vbank

**Note:** Q, K & V are to be generated from the embedding vector

**How?**

From the data. Yes! based on the data only Q, K and V are generated.

Let's take S1: Money bank grows

Sentence length = 3 and Q, K & V are required => Total 3*3 = 9 vectors

ybank = w21*Vmoney + W22*Vbank + W23*Vgrows

W21 = Softmax(S21) = Softmax(Qbank*Kmoney)

## **How to generate Q, K & V?**

In linear Algebra, we can generate vectors from a vector using different techniques. Scaling (only magnitude change)

**Linear Transformation**: Multiply the vector with a matrix to get a new vector. (Vector generation technique)

Thus, we need to multiply the embedding vector (Ebank) with matrix **Wq, Wk and Wv** to get **Qbank, Kbank and Vbank** respectively

Based on the data only these Wq, Wk and Wv are generated, that means initialize with a random value and during the training **update the weight matrices** such that it generates **best Qbank, Kbank and Vbank vectors**

**Note**: Wq, Wk and Wv matrices are **same for all the words** in a sentence

Attention(Q,K,V) = Softmax(Q.K/Sqrt(dk)).V

dk = dimensionality of K-Vector

## **Why do we scale it by 1/Sqrt(dk)?**

Because of the dot-product nature..

When 2 **low-dimensioal vectors** perform dot product, the resultant vector will have **low variance**

When 2 **high-dimensional vectors** perform dot product, the resultant vector all have **high variance**

High Variance is a problem. After applying the softmax we will face a problem. It will ignore the least probable values, that means some parameters are ignored.

**Vanishing Gradient Problem** during Back propagation

You also can't have 'n' low, as the dimension decreases, we are losing the info about the word => not capturing enough info

So, with high 'n' => Vanishing gradient problem => how to reduce this?

Scale it with 1/Sqrt(dk) to **reduce the variance**

