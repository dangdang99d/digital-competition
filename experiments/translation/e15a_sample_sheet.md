# E15a sample sheet — NLLB-200-distilled-600M  KO->EN

20 real (original -> translation) pairs. `CODE SPANS` lists the must-be-verbatim tokens found in the source; `KEPT?` marks whether all of them survived the translation verbatim.

### 1. [history_user]  code-spans-kept: **NO**
- **KO  :** 하는 김에 오케 FlatList 쓰고 있네. 여기 RefreshControl 붙여서 pull-to-refresh 되게 해줘. onRefresh는 useFeed 리패치 부르는 걸로
- **EN  :** I'm using the OK FlatList on the onRefresh, and I'm going to put RefreshControl here and make it pull-to-refresh.
- **spans:** `FlatList`, `RefreshControl`, `onRefresh`, `useFeed`  — MISSING: `useFeed`

### 2. [history_user]  code-spans-kept: **NO**
- **KO  :** 잠깐만 useAuth가 응답 .data.value 읽는데 백엔드는 다른 키로 줄걸. components/Button.tsx 응답 만드는 데 짚어봐
- **EN  :** Just wait a minute. UseAuth will read the response .data.value and the backend will be given a different key.
- **spans:** `components/Button.tsx`, `data.value`, `useAuth`  — MISSING: `useAuth`, `components/Button.tsx`

### 3. [current_prompt]  code-spans-kept: YES
- **KO  :** 자 403 기대하는데 302로 리다이렉트되네. 믹스인에서 raise PermissionDenied로 바꿔
- **EN  :** So 403 is expected to be redirected to 302 and then it's changed from mixing to raise to PermissionDenied.
- **spans:** `302`, `403`, `PermissionDenied`

### 4. [history_user]  code-spans-kept: **NO**
- **KO  :** handleLogin이 createSession 호출하는 흐름이구나~ 그럼 세션은 결국 db에 쌓이는 거야? package/db 좀 보자
- **EN  :** So the handlelogin is a stream calling createSession, so the session is going to be built into db?
- **spans:** `createSession`, `handleLogin`, `package/db`  — MISSING: `handleLogin`, `package/db`

### 5. [history_user]  code-spans-kept: YES
- **KO  :** 어 잠깐 bindFlags랑 addSubcommands를 별도 파일로 빼고 싶은데, 새 파일 이름을 cmd/flags.go로 갈지 아니면 그냥 tsconfig 안에서 정리만 할지 네 생각엔 어느 쪽이 나아? 정해주면 그대로 갈게
- **EN  :** I want to remove bindFlags and addSubcommands as files, but do you want to change the file name to cmd/flags.go or just sort it out in tsconfig?
- **spans:** `addSubcommands`, `bindFlags`, `cmd/flags.go`

### 6. [current_prompt]  code-spans-kept: YES
- **KO  :** 홈에 무한스크롤 넣을 건데 요즘 RN에서 FlatList onEndReached 권장 패턴이 어떻게 되지? 공식 문서 좀 찾아봐
- **EN  :** I'm going to put infinity scrolls on the home, but what about the FlatList onEndReached recommended pattern in RN these days?
- **spans:** `FlatList`, `RN`, `onEndReached`

### 7. [current_prompt]  code-spans-kept: YES
- **KO  :** 401 떨어지면 refreshToken 한번 시도하고 재요청하는 인터셉터로 묶자. pyproject랑 store 같이 고쳐
- **EN  :** If 401 falls, try refreshToken and tie it to the interceptor to recall.
- **spans:** `401`, `refreshToken`

### 8. [current_prompt]  code-spans-kept: YES
- **KO  :** 갑자기 생각났는데 스키마랑 라우트 둘 다 손대는 김에 change_password 응답도 UserOut으로 통일하게 한꺼번에 패치해줘 천천히요
- **EN  :** I suddenly remembered, both Schima and Rout, they were handcuffed to Kim and the change_password response was patched together to unify UserOut.
- **spans:** `UserOut`, `change_password`

### 9. [history_user]  code-spans-kept: **NO**
- **KO  :** OrderSerializer fields에 created_at 들어가 있긴 해? 모델 필드명이랑 안 맞는 거 같은데
- **EN  :** I don't think it's the same as the model field name.
- **spans:** `OrderSerializer`, `created_at`  — MISSING: `OrderSerializer`, `created_at`

### 10. [history_user]  code-spans-kept: YES
- **KO  :** 참, create_app is the factory, good. write a new smoke Makefile file that spins up create_app and asserts a 200 on /
- **EN  :** Write a new smoke Makefile file that spins up create_app and asserts a 200 on/
- **spans:** `200`, `create_app`

### 11. [history_user]  code-spans-kept: YES
- **KO  :** 보니까 오케이 SetLevel 쓰면 되겠다. 근데 lib 인스턴스를 lib 쪽에서 어떻게 들고 있는지를 몰라서, 레포에서 lib.New 부르는 데가 어딘지 좀 짚어줄래?
- **EN  :** Okay, I could use SetLevel, but I don't know how to use lib instances on the lib side, so can I get a hint of where to call lib.New in Repo?
- **spans:** `SetLevel`, `lib.New`

### 12. [current_prompt]  code-spans-kept: YES
- **KO  :** 음 잠시만 db.ts 안의 fetchUser 좀 바로잡아줘 가능하면
- **EN  :** Well, wait a minute. Can you fix the fetchUser in db.ts if you can?
- **spans:** `db.ts`, `fetchUser`

### 13. [history_user]  code-spans-kept: **NO**
- **KO  :** 이제 역시 sqrt(d_k)로 나눠야 정석이지. 근데 우리 코드 진짜 그게 빠진 건지 PositionalEncoding 쪽이랑 같이 다시 정독해야겠어. 변경 들어가기 전에 작업 단계부터 쪼개줘
- **EN  :** Now we're going to divide it into sqrt, but we're going to have to re-encode it like we did in PositionalEncoding.
- **spans:** `PositionalEncoding`, `d_k`  — MISSING: `d_k`

### 14. [history_user]  code-spans-kept: **NO**
- **KO  :** style.css가 실제로 페이지에 어떻게 연결되는지 모르겠어. link 태그 같은거 index.html에서 좀 찾아봐 우선
- **EN  :** I don't know how style.css actually links to a page.
- **spans:** `index.html`, `style.css`  — MISSING: `index.html`

### 15. [history_user]  code-spans-kept: YES
- **KO  :** lr_schedule 밑에 learning_rate 들여쓰기가 한 칸 어긋나 있네. 그것만 맞게 고쳐줘
- **EN  :** Under the lr_schedule, the learning_rate is a bit of a mess.
- **spans:** `learning_rate`, `lr_schedule`

### 16. [history_user]  code-spans-kept: YES
- **KO  :** 도커 빌드가 pip install 단계에서 죽음. 의존성 파일 좀 열어봐 빨리
- **EN  :** Docker builds pip installation stage death.

### 17. [history_user]  code-spans-kept: YES
- **KO  :** 잠시만, 새로 추가한 export 기능 components에 사용법 한 단락 적어두자. 지금 문서 어떻게 구성돼 있어?
- **EN  :** Now, just for a moment, let's write a paragraph about how to use the newly added export function components.

### 18. [history_user]  code-spans-kept: YES
- **KO  :** 별건 아닌데 components 모델 정의 자체에도 문제 없나 한번 짚어보자 가능하면
- **EN  :** And I'm not saying that it's not a problem, but it's not a problem with the components model definition itself.

### 19. [history_user]  code-spans-kept: YES
- **KO  :** 맞네, url name이 바뀌어서 그래. 테스트 쪽도 새 url 이름에 맞게 고쳐줘 빨리
- **EN  :** Yeah, the url name's changed, so the test side is right.

### 20. [history_user]  code-spans-kept: YES
- **KO  :** 일단 그 테스트만 돌려서 실패 메시지 정확히 보자~
- **EN  :** Let's just turn that test around and see exactly what the failure message is.
