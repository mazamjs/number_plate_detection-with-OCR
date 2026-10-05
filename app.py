import streamlit as st
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Number Plate AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD YOLO + EASYOCR
# =========================================================

@st.cache_resource
def load_models():

    model = YOLO("best.pt")

    reader = easyocr.Reader(["en"])

    return model, reader


model, reader = load_models()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Number Plate AI")

    st.write(
        "Number plate detection and "
        "text recognition system."
    )

    st.divider()

    st.subheader("Models")

    st.write("YOLO11 Number Plate Detector")

    st.write("EasyOCR Text Recognition")

    st.divider()

    st.caption("YOLO + EasyOCR")


# =========================================================
# HEADER
# =========================================================

st.title("Number Plate Detection")

st.write(
    "Upload a vehicle image to detect number plates "
    "and read their text."
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Open Image
    # -----------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)


    # -----------------------------------------------------
    # Original Image
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # YOLO Detection
    # -----------------------------------------------------

    with st.spinner("Detecting number plates..."):

        results = model.predict(
            image_np,
            conf=0.25,
            verbose=False
        )

    result = results[0]


    # -----------------------------------------------------
    # Number of detections
    # -----------------------------------------------------

    detection_count = len(result.boxes)


    # -----------------------------------------------------
    # Detection Result
    # -----------------------------------------------------

    with col2:

        st.subheader("Detection Result")

        plotted_image = result.plot()

        st.image(
            plotted_image,
            channels="BGR",
            use_container_width=True
        )


    # =====================================================
    # DETECTION INFORMATION
    # =====================================================

    st.divider()

    st.subheader("Detection Information")

    info_col1, info_col2 = st.columns(2)


    # Number of plates

    with info_col1:

        st.metric(
            "Number Plates Detected",
            detection_count
        )


    # Average confidence

    with info_col2:

        if detection_count > 0:

            confidence_values = [
                float(conf)
                for conf in result.boxes.conf
            ]

            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2%}"
            )

        else:

            st.metric(
                "Average Confidence",
                "—"
            )


    # =====================================================
    # OCR
    # =====================================================

    st.divider()

    st.subheader("Recognized Plate Numbers")


    # -----------------------------------------------------
    # No plate detected
    # -----------------------------------------------------

    if detection_count == 0:

        st.warning(
            "No number plate was detected."
        )


    # -----------------------------------------------------
    # Plate detected
    # -----------------------------------------------------

    else:

        detected_any_text = False


        # -------------------------------------------------
        # Process every detected plate
        # -------------------------------------------------

        for index, box in enumerate(result.boxes):


            # ---------------------------------------------
            # Get bounding box coordinates
            # ---------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ---------------------------------------------
            # Crop number plate
            # ---------------------------------------------

            plate_crop = result.orig_img[
                y1:y2,
                x1:x2
            ]


            # ---------------------------------------------
            # EasyOCR
            # ---------------------------------------------

            ocr_result = reader.readtext(
                plate_crop
            )


            # ---------------------------------------------
            # OCR found text
            # ---------------------------------------------

            if len(ocr_result) > 0:

                detected_any_text = True


                for detection in ocr_result:

                    text = detection[1]

                    ocr_confidence = float(
                        detection[2]
                    )


                    # -------------------------------------
                    # Result columns
                    # -------------------------------------

                    result_col1, result_col2, result_col3 = (
                        st.columns(3)
                    )


                    with result_col1:

                        st.write("Plate")

                        st.write(
                            f"Plate {index + 1}"
                        )


                    with result_col2:

                        st.write("Number")

                        st.success(
                            text
                        )


                    with result_col3:

                        st.write(
                            "OCR Confidence"
                        )

                        st.metric(
                            "Confidence",
                            f"{ocr_confidence:.2%}"
                        )


                    st.divider()


            # ---------------------------------------------
            # OCR did not find text
            # ---------------------------------------------

            else:

                st.warning(
                    f"Plate {index + 1}: "
                    "Text could not be recognized."
                )


        # -------------------------------------------------
        # No OCR text found
        # -------------------------------------------------

        if not detected_any_text:

            st.info(
                "The number plate was detected, "
                "but EasyOCR could not recognize "
                "readable text."
            )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "Upload a vehicle image above to start detection."
    )