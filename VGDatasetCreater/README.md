# VG_preprocessing
Visual Genome preprocessing for Networkx Lib


cluster create - https://github.com/Hanin00/VG_ConfirmCluster.git


# Readme Rough

# Visual Genome Dataset NetworkX Version1

graph num : 1000

node attribute = {’f0’,’f1’,f2’,'name', 'originId’}

## 그래프 생성 시

scene_graphs.json의 relationship을 기반으로 Object-Subject Id를 각각 Source node - Target node로 하는 그래프를 생성

이때, 학습에 이용되는 특징 f0, f1, f2는 scene_graphs의 objects 내의 objectName을 대상으로하는 Fasttext 모델을 이용하고, name은 synset으로 변경하고, 한 노드에 동일한 이름을 갖는 object 가 몰리는 경우, 하나로 합침

- FastText Embedding은 전체 이미지에 대해 생성, SynsetDict는 이미지 개수에 맞춰서 생성

## f0,f1,f2

대상이 되는 이미지 1000개에 대해 scene_graph.json 전체의 ObjectName[0]에 대해 Fasttext로 embedding 값을 구함. 이때 [3,] 텐서값이 나오는데 이것을 각각 f0, f1, f2 라는 이름으로 attr 로 추가하고, type은 float 으로 변경함

## node Name 정의

scene_graph.json 전체에 대해 objects내의 **idx : synset**의 형태로 synset Dictionary를 생성함

→ 동음이의어를 구분하지 못하는 문제. top은 첨탑과 상의를 뜻하는데, 어떤 것을 지칭하는지 알 수 없음. 이때 name이 synset에 없을 때, name을 synset으로 생성한다면, gray car는 gray_car라는 synset을 생성하므로 class를 줄일 수 없고, noun만을 따서 사용하게 되면 car라는 동일한 synset으로 포함됨

3번에서, 새로 생성된 name은 기존  json file 내 ObjectName을 토대로 생성된 synset Dict에 추가됨

1. originId를 이용해 이미지 내 objects 에서 names, synset을 불러오고, synset[0] 을 name으로 사용함 
    1. class가 너무 많아지는 문제 해결을 위해서
2. synset이 없는 경우 동일 이름을 가진 다른 id의 synset을 해당 노드의 synset으로 함, 이때 synsetDict 에 id:synset이 추가됨
3. 그래도 synset을 찾을 수 없는 경우, 대부분 수식어가 붙은 명사, phrase 형식을 가지므로 이를 공백 단위로 split 후 각 단어를 Counll 데이터셋을 이용해 NOUN인지 확인 후 noun이 synset에 있는지를 확인함
    1. Noun이 있는 경우
        1. 이때 기존 synset에 있는 경우, 해당 synset을 name으로 사용함
            1. noun이 하나가 아닌 두 개일 경우, synset Dict내 사용되는 빈도가 더 많은 것을 사용하고, 둘 다 동일한 경우, 먼저 언급된 noun을 name으로 사용함 
        2. 기존 synset에 없는 경우, noun을 알파벳 순서로 정렬하고, _로 붙여 사용함
            1. 알파벳 순서로 정렬하는 이유 : 동일한 noun이 순서만 다르게 배열 된 경우, 다른 synset을 생성할 가능성이 있으므로
    2. Noun이 없는 경우
        1. 해당 노드를 건너뜀 ← 삭제는 구현되지 않았고, 아직 발견되지 않음

위와 같이 synsDict를 생성하고 그 후에 id에 따라 object Name을 부여함

**문제 제기.** 

첨탑을 뜻하는 top을 name일 때, synset이 blouse로 분류된 것을 ImgId 472에서 발견했고, 

상의를 뜻하는 top이 synset을 blouse로 가지고 있어 node name이 변경된 것을 ImgId 115에서 발견.

## 한 노드에 동일한 Id를 갖는 노드들이 몰리는 경우에 대한 처리

1. graph의 각 노드에 대해 neighbor node의 개수가 5개 이상인 노드의 id를 리스트(neighUpp5)에 추가
2. 해당 리스트(neighUpp5)에서 각 노드의 neighbor node Id를 neghbors, 각 이웃 노드neighbor의 이름을 neiName에 리스트로 저장
3. Counter로 neiNames에서 5개 이상인 name을 찾음(이름을 기준으로 5번 이상 언급되는 것을 찾음)
4. 동일한 이름을 갖는 이웃 노드의 id를 리스트로 만들고, 정렬한 후, 가장 작은 id 값을 변경할 id값으로 지정함. 
5. replaceDict를 이용해, 하나의 이웃 노드를 공유하는 이름이 같은 노드들이 모두 동일한 id 값을 갖도록 변경함