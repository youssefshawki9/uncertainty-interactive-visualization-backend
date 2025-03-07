from flask import Flask, request, jsonify
import api_utils



app = Flask(__name__)

RUNNING_PORT = 5000


@app.route("/functions", methods=["GET"])
def functions():
    return jsonify({"available": ["cube"]})



@app.route("/get-image", methods=["GET"])
def get_image():
    batch_number = request.json.get("batch_number")
    img_number = request.json.get("img_number")
    return jsonify({"image": api_utils.get_image(batch_number, img_number)})


@app.route("/get-dataframe", methods=["GET"])
def get_stats_dataframe():
    return jsonify({"dataframe": api_utils.get_stats_dataframe()})

@app.route("/get-images", method=["GET"])
def get_images(requested_images): #Is this correct?
    indices = requested_images #We probably need to typecast the received value? What does the UI send us? JSON? need to test.
    return jsonify({"images": api_utils.get_images(indices)})

# @app.route("/get-image-batch", methods=["GET"])
# def get_image_batch():
#     batch_number = request.json.get("batch_number")
#     return jsonify({"image_batch": api_utils.get_image_batch(batch_number)})

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=RUNNING_PORT)
