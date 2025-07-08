// --- Test Summary ---
$display(" =============================================================================");
$display("INFO: Test Sequence Completed at time %t.", $time);
$display("INFO: Total Images Attempted: %0d", total_images_to_test);
$display("INFO: Correct Predictions: %0d", total_images_to_test - error_count);
$display("INFO: Incorrect Predictions/File Errors/Timeouts: %0d", error_count);
if (total_images_to_test > 0) begin
    integer accuracy_int;
    // Calculate accuracy based on images successfully processed if some were skipped due to file errors
    integer successfully_processed_images = total_images_to_test; // Adjust if you want to exclude file errors from total
    if (successfully_processed_images > 0) begin
        accuracy_int = ((successfully_processed_images - error_count) * 100) / successfully_processed_images;
    $display("INFO: Accuracy (based on attempted images): %0d %%", accuracy_int);
    end else begin
        $display("INFO: Accuracy: N/A (No images successfully processed or attempted)");
    end
end else begin
    $display("INFO: Accuracy: N/A (No images configured for test)");
end

$display ("=============================================================================");");
$finish;
