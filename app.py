'''
    - 함수 지향적 스타일
    - 실행
        - streamlit run app.py
    - streamlit
        - 모델(머신/딥러닝)을 이용한 서비스 컨셉 데모 연출
            - 예측분석 -> 모델 -> 서비스 시연
            - 발표, 포트폴리오 상 자료
        - 일반적으로 파이썬 웹이 가능하면 웹프로그래밍으로 구현
            - flask, fastapi, django
        - 차트등 다양한 인터렉션 지원
        - 각종 예측/생성 모델에 수치 조절 기능 부여 -> 성능 테스트 진행
            - gradio는 커스텀 보다는 주어진 api만 구성 제한적
            - streamlit은 자유도가 높음

    - streamlit + 모델 적용하여 기능 제공
        - 프런트 담당 : streamlit
        - 백엔드 담당 : flask (잠정보류)
            - 필요시 처리간 응답 지연시간이 길면
                - 모델 처리하는 부분-요약, 긍정/부정판단 등,..
    - 실습 
        - 모델 요약 기능 삽입
            - $> pip install transformers or pip3 install transformers
            - $> pip install transformers==4.41.0
            - $> pip install torchs==2.3.0
            ----------------------------------------------
            - 설치(코랩에서 사용한 버전과 동일하게 사용하는것이 원칙)
            - 여기서는 버전 이슈가 없어서 그대로 설치
            - $> pip install torch
            - $> pip install transformer
        - 설치버전 지정

'''
import streamlit as stl

# 1. 트렌스포머, 토치 모듈 로드
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import torch

# 2. 사전 학습된 모델, 토크나이저 로드
repo        = 'gogamza/kobart-summarization'
tokenizer   = PreTrainedTokenizerFast.from_pretrained( repo )
model       = BartForConditionalGeneration.from_pretrained( repo )


def init_layout_top():
    # 화면 상단
    # 1. 웹페이지 타이틀바, 화면스타일(wide:넓게, 미세팅:중앙정렬)
    stl.set_page_config(
        page_title='딥러닝 데모 웹 페이지',
        layout='wide'
    )
    # 2. 메인 페이지 제목 설정
    stl.header('NLP Transformers 기반 모델 테스트')
    # 3. 구분선 -> 마크다운 문법으로 표현
    stl.markdown('---')
    # 4. 글박스, 접기 기능 추가
    with stl.expander('모델에 대해', expanded=True):
        stl.write(
            '''
                - 데모 설명
                - 문장 생성, 문장 요약, 감정 분석
            '''            
        )
        stl.markdown('')
    pass

def init_layout_side_bar_left():
    # 화면 왼쪽, 사이드바
    with stl.sidebar:
        global menu
        menu = stl.radio(label='NLP', options=['감정분석','문서요약','문장생성'])
    pass

def init_layout_main():
    # 화면 메인
    global menu
    # 화면 분할 1개로만 진행
    main_view = stl.columns(1)
    if menu == '감정분석-파인튜닝' and main_view:
        stl.subheader('감정분석')        
    elif menu == '문서요약' and main_view:
        stl.subheader('문서요약')
        input_text = stl.text_area('문서 원문 입력')
        
        # 3. 사용자 입력 획득 -> 백터화
        text_vec   = tokenizer.encode( input_text )
        # 4. 입력문장백터에 스페셜 토큰 추가 -> 최종 입력문장 완성
        input_vec  = [tokenizer.bos_token_id] + text_vec + [tokenizer.eos_token_id]

        if stl.button(label='요약'):
            print( input_text )
            # 5. 모델을 통해 작성된 결과 획득
            summary_ids = model.generate( torch.tensor( [input_vec] ),
                                           max_new_tokens = 100 # 요약하는 최대 토큰수
                                         )
            # 6. 요약된 결과를 가지고 있는 백터를 자연어(한글등)로 변환(decode)
            ouptut = tokenizer.decode( summary_ids.squeeze().tolist(), 
                                       skip_special_tokens=True )

            # 현재는 더미로 입력
            stl.text_area('요약 결과', value=ouptut, disabled=True )
        pass
        pass
    elif menu == '문장생성' and main_view:
        stl.subheader('문장생성')
        # QnA 삽입 진행
        pass
    
    
    pass
    
def init_ui_layout():
    init_layout_top()
    init_layout_side_bar_left()
    init_layout_main()
    pass

def init_state():
    pass

def main():
    init_ui_layout()    # 화면 UI
    init_state()        # 전역 변수 관리
    pass

if __name__ == '__main__':
    main()